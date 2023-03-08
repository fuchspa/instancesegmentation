import tensorflow as tf


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
