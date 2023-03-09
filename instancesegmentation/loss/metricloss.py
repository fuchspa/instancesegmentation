import tensorflow as tf


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
        name: str = "MetricLoss",
        **kwargs,
    ) -> None:
        super().__init__(trainable=False, name=name, **kwargs)

    def call(self, embeddings: tf.Tensor, labels: tf.Tensor) -> tf.Tensor:
        pass
