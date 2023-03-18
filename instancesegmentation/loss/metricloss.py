import tensorflow as tf

from instancesegmentation.loss.distancemetric import (
    DistanceMetric,
    ManhattanDistanceMetric,
)
from instancesegmentation.loss.errormetric import (
    ErrorMetric,
    SquaredErrorMetric,
)

# It is okay to use these as defaults, as they have no internal state
_default_distance_metric = ManhattanDistanceMetric()
_default_error_metric = SquaredErrorMetric()


def strictly_upper_triangular_matrix(x: tf.Tensor) -> tf.Tensor:
    """Return the strictly upper triangular matrix of a given matrix."""
    x = tf.linalg.band_part(x, 0, -1)
    x = tf.linalg.set_diag(x, tf.zeros((x.shape[0],)))
    return x


@tf.function
def compute_centroids(
    embeddings: tf.Tensor, labels: tf.Tensor
) -> tuple[tf.Tensor, tf.Tensor]:
    """
    Compute the centroids of each label.

    Expects embeddings and labels in flattened form, i.e. (n, feature_count) and (n,). Emebddings
    are expected to be floating point values, while the labels are integer values.
    """
    number_of_instances = tf.math.reduce_max(labels) + 1
    centroids = tf.TensorArray(tf.float32, size=number_of_instances)
    instance_sizes = tf.TensorArray(tf.int32, size=number_of_instances)
    for i in tf.range(number_of_instances):
        mask = labels == i
        instance_size = tf.reduce_sum(tf.cast(mask, tf.int32))
        centroid = (
            tf.math.reduce_mean(embeddings[mask], axis=0)
            if instance_size > 0
            else tf.zeros_like(embeddings[0, :])
        )
        centroids = centroids.write(i, centroid)
        instance_sizes = instance_sizes.write(i, instance_size)
    return centroids.stack(), instance_sizes.stack()


class MetricLoss(tf.keras.layers.Layer):
    """
    Compute sample-wise push and pull losses.

    It pulls the computed features of the same class closer together and the centroids of different
    classes farther apart. To prevent moving the centroids towards infinity a regularization can be
    added.
    """

    def __init__(
        self,
        push_distance_metric: DistanceMetric = _default_distance_metric,
        pull_distance_metric: DistanceMetric = _default_distance_metric,
        push_error_metric: ErrorMetric = _default_error_metric,
        pull_error_metric: ErrorMetric = _default_error_metric,
        push_margin: float = 0.25,
        pull_margin: float = 0.0,
        push_weight: float = 1.0,
        pull_weight: float = 1.0,
        regularization_error_metric: ErrorMetric = _default_error_metric,
        regularization_weight: float = 1e-4,
        name: str = "MetricLoss",
        **kwargs,
    ) -> None:
        super().__init__(trainable=False, name=name, **kwargs)
        self.push_distance_metric = push_distance_metric
        self.pull_distance_metric = pull_distance_metric
        self.push_error_metric = push_error_metric
        self.pull_error_metric = pull_error_metric
        self.push_margin = push_margin
        self.pull_margin = pull_margin
        self.push_weight = push_weight
        self.pull_weight = pull_weight
        self.regularization_error_metric = regularization_error_metric
        self.regularization_weight = regularization_weight

    def _compute_push_loss(self, centroids: tf.Tensor) -> tf.Tensor:
        """Push apart the different cluster centroids."""
        number_of_instances = tf.cast(tf.shape(centroids)[0], tf.float32)
        if number_of_instances < 2:
            return tf.constant(0.0)
        number_of_comparisons = number_of_instances * (number_of_instances - 1.0) / 2.0
        centroids = tf.tile(centroids[:, None, :], [1, centroids.shape[0], 1])
        distances = self.push_distance_metric(
            centroids, tf.transpose(centroids, perm=(1, 0, 2))
        )
        distances = tf.nn.relu(self.push_margin - distances)
        distances = strictly_upper_triangular_matrix(distances)
        loss = self.push_error_metric(distances)
        return tf.math.reduce_sum(loss) / number_of_comparisons

    def _compute_pull_loss(
        self, embeddings: tf.Tensor, centroids: tf.Tensor, labels: tf.Tensor
    ) -> tf.Tensor:
        """Pull the members of a cluster closer together."""
        centroids = tf.gather(centroids, labels)
        distances = self.pull_distance_metric(embeddings, centroids)
        distances = tf.nn.relu(distances - self.pull_margin)
        loss = self.pull_error_metric(distances)
        return tf.math.reduce_mean(loss)

    def _compute_regularization_loss(self, centroids: tf.Tensor) -> tf.Tensor:
        """Compute the distance to zero to prevent centroids from drifting too far away."""
        loss = self.regularization_error_metric(centroids)
        return tf.math.reduce_mean(loss)

    def call(self, embeddings: tf.Tensor, labels: tf.Tensor) -> tf.Tensor:
        embedding_losses = list()
        for embedding, instance_label in zip(embeddings, labels):
            instance_label = tf.reshape(instance_label, (-1,))
            embedding = tf.reshape(embedding, (-1, embedding.shape[-1]))
            centroids, instance_sizes = compute_centroids(embedding, instance_label)
            valid_centroids = tf.gather(
                centroids, tf.squeeze(tf.where((instance_sizes > 0)), axis=-1)
            )

            push_loss = self._compute_push_loss(valid_centroids)
            pull_loss = self._compute_pull_loss(embedding, centroids, instance_label)
            regularization_loss = self._compute_regularization_loss(valid_centroids)

            embedding_losses.append(
                self.push_weight * push_loss
                + self.pull_weight * pull_loss
                + self.regularization_weight * regularization_loss
            )

        return tf.reduce_mean(embedding_losses)
