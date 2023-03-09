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
_default_distance_measure = ManhattanDistanceMetric()
_default_error_measure = SquaredErrorMetric()


def strictly_upper_triangular_matrix(x: tf.Tensor) -> tf.Tensor:
    """Return the upper triangular matrix of a given matrix."""
    x = tf.linalg.band_part(x, 0, -1)
    x = tf.linalg.set_diag(x, tf.zeros((x.shape[0],)))
    return x


def compute_centroids(embeddings: tf.Tensor, labels: tf.Tensor) -> tf.Tensor:
    """
    Compute the centroids of each label.

    Expects embeddings and labels in flattened form, i.e. (n, feature_count) and (n,). Emebddings
    are expected to be floating point values, while the labels are integer values.
    """
    centroids = list()
    number_of_instances = tf.math.reduce_max(labels)
    for i in range(int(number_of_instances) + 1):
        mask = labels == i
        instance_size = tf.reduce_sum(tf.cast(mask, tf.float32))
        if instance_size < 1e-6:
            continue
        centroid = tf.math.reduce_mean(embeddings[mask], axis=0)
        centroids.append(centroid)
    return tf.stack(centroids)


class MetricLoss(tf.keras.layers.Layer):
    """
    Compute sample-wise push and pull losses.

    It pulls the computed features of the same class closer together and the centroids of different
    classes farther apart. To prevent moving the centroids towards infinity a regularization can be
    added.
    """

    def __init__(
        self,
        push_distance_metric: DistanceMetric = _default_distance_measure,
        push_error_metric: ErrorMetric = _default_error_measure,
        push_margin: float = 0.25,
        name: str = "MetricLoss",
        **kwargs,
    ) -> None:
        super().__init__(trainable=False, name=name, **kwargs)
        self.push_distance_metric = push_distance_metric
        self.push_error_metric = push_error_metric
        self.push_margin = push_margin

    def _compute_push_loss(self, centroids: tf.Tensor) -> tf.Tensor:
        """Pull apart the different cluster centroids."""
        number_of_instances = tf.cast(tf.shape(centroids)[0], tf.float32)
        number_of_comparisons = number_of_instances * (number_of_instances - 1.0) / 2.0
        centroids = tf.tile(centroids[:, None, :], [1, centroids.shape[0], 1])
        distances = self.push_distance_metric(
            centroids, tf.transpose(centroids, perm=(1, 0, 2))
        )
        distances = tf.nn.relu(self.push_margin - distances)
        distances = strictly_upper_triangular_matrix(distances)
        loss = self.push_error_metric(distances)
        return tf.math.reduce_sum(loss) / number_of_comparisons

    def call(self, embeddings: tf.Tensor, labels: tf.Tensor) -> tf.Tensor:
        embedding_losses = list()
        for embedding, instance_label in zip(embeddings, labels):
            instance_label = tf.cast(tf.reshape(instance_label, -1), tf.int32)
            embedding = tf.reshape(embedding, (-1, embedding.shape[-1]))
            centroids = compute_centroids(embedding, instance_label)

            push_loss = self._compute_push_loss(centroids)

            embedding_losses.append(push_loss)

        return tf.reduce_mean(embedding_losses)
