import tensorflow as tf


def _build_encoding_step(
    filters: int, name: str, downsample: bool = True
) -> list[tf.keras.layers.Layer]:
    layers = list()
    if downsample:
        layers.append(
            tf.keras.layers.Conv2D(
                filters=filters,
                kernel_size=3,
                strides=2,
                padding="same",
                activation=tf.keras.layers.LeakyReLU(),
                kernel_regularizer=tf.keras.regularizers.L2(1e-4),
                name=f"{name}_downsample",
            )
        )
    layers.append(
        tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=3,
            padding="same",
            activation=tf.keras.layers.LeakyReLU(),
            kernel_regularizer=tf.keras.regularizers.L2(1e-4),
            name=name,
        )
    )
    return layers


def _build_decoding_step(
    filters: int, name: str, upsample: bool = True
) -> list[tf.keras.layers.Layer]:
    layers = list()
    if upsample:
        layers.append(
            tf.keras.layers.Conv2DTranspose(
                filters=filters,
                kernel_size=4,
                strides=2,
                padding="same",
                activation=tf.keras.layers.LeakyReLU(),
                kernel_regularizer=tf.keras.regularizers.L2(1e-4),
                name=f"{name}_upsample",
            )
        )
    layers.append(
        tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=3,
            padding="same",
            activation=tf.keras.layers.LeakyReLU(),
            kernel_regularizer=tf.keras.regularizers.L2(1e-4),
            name=name,
        )
    )
    return layers


def _build_embedding_head(
    number_of_embeddings: int, name: str = "embedding"
) -> tf.keras.layers.Layer:
    return tf.keras.layers.Conv2D(
        filters=number_of_embeddings,
        kernel_size=3,
        padding="same",
        activation=None,
        kernel_regularizer=tf.keras.regularizers.L2(1e-4),
        name=name,
    )


def _build_segmentation_head(
    number_of_classes: int, name: str = "segmentation"
) -> tf.keras.layers.Layer:
    return tf.keras.layers.Conv2D(
        filters=number_of_classes,
        kernel_size=3,
        padding="same",
        activation=None,
        kernel_regularizer=tf.keras.regularizers.L2(1e-4),
        name=name,
    )


class EmbeddingModel(tf.keras.models.Model):
    """
    A simple encoder-decoder-model with two heads to learn some embeddings and a segmentation on the dummy data.

    First output are the embedding features.
    Second output is the semantic segmentation in foreground/background.
    """

    def __init__(
        self, number_of_embeddings: int = 8, number_of_classes: int = 2
    ) -> None:
        super().__init__()

        self.layers_down = list()
        for i, (filters, downsample) in enumerate(
            zip([16, 32, 64, 128], [False, True, True, True])
        ):
            self.layers_down.extend(
                _build_encoding_step(filters, f"encoding_{i + 1}", downsample)
            )

        self.global_information = tf.keras.layers.GlobalAveragePooling2D(
            keepdims=True,
        )

        self.layers_up = list()
        for i, (filters, upsample) in enumerate(
            zip([128, 64, 32, 16], [False, True, True, True])
        ):
            self.layers_up.extend(
                _build_decoding_step(filters, f"decoding_{i + 1}", upsample)
            )

        self.embedding = _build_embedding_head(number_of_embeddings)
        self.segmentation = _build_segmentation_head(number_of_classes)

    def call(self, x: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        for layer in self.layers_down:
            x = layer(x)

        global_features = self.global_information(x)
        x = tf.nn.relu(x + global_features)

        for layer in self.layers_up:
            x = layer(x)

        embedding = self.embedding(x)
        segmentation = self.segmentation(x)

        return embedding, segmentation
