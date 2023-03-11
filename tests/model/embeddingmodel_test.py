import tensorflow as tf

from instancesegmentation.model.embeddingmodel import (
    _build_decoding_step,
    _build_embedding_head,
    _build_encoding_step,
    _build_segmentation_head,
    EmbeddingModel,
)


def test_build_encoding_step() -> None:
    layers = _build_encoding_step(filters=16, name="encoding_1", downsample=False)
    assert len(layers) == 1
    assert isinstance(layers[0], tf.keras.layers.Conv2D)
    assert layers[0].filters == 16
    assert layers[0].name == "encoding_1"
    assert layers[0].strides == (1, 1)

    layers = _build_encoding_step(filters=32, name="encoding_2", downsample=True)
    assert len(layers) == 2
    assert isinstance(layers[0], tf.keras.layers.Conv2D)
    assert layers[0].filters == 32
    assert layers[0].name == "encoding_2_downsample"
    assert layers[0].strides == (2, 2)
    assert isinstance(layers[1], tf.keras.layers.Conv2D)
    assert layers[1].filters == 32
    assert layers[1].name == "encoding_2"
    assert layers[1].strides == (1, 1)


def test_build_decoding_step() -> None:
    layers = _build_decoding_step(filters=128, name="decoding_1", upsample=False)
    assert len(layers) == 1
    assert isinstance(layers[0], tf.keras.layers.Conv2D)
    assert layers[0].filters == 128
    assert layers[0].name == "decoding_1"
    assert layers[0].strides == (1, 1)

    layers = _build_decoding_step(filters=64, name="decoding_2", upsample=True)
    assert len(layers) == 2
    assert isinstance(layers[0], tf.keras.layers.Conv2DTranspose)
    assert layers[0].filters == 64
    assert layers[0].name == "decoding_2_upsample"
    assert layers[0].strides == (2, 2)
    assert isinstance(layers[1], tf.keras.layers.Conv2D)
    assert layers[1].filters == 64
    assert layers[1].name == "decoding_2"
    assert layers[1].strides == (1, 1)


def test_build_embedding_head() -> None:
    number_of_embeddings = 8
    embedding = _build_embedding_head(number_of_embeddings)

    assert isinstance(embedding, tf.keras.layers.Conv2D)
    assert embedding.filters == number_of_embeddings
    assert embedding.name == "embedding"
    assert embedding.activation == tf.keras.activations.linear


def test_build_segmentation_head() -> None:
    number_of_classes = 2
    segmentation = _build_segmentation_head(number_of_classes)

    assert isinstance(segmentation, tf.keras.layers.Conv2D)
    assert segmentation.filters == number_of_classes
    assert segmentation.name == "segmentation"
    assert segmentation.activation == tf.keras.activations.linear


def test_embedding_model_smoke_test() -> None:
    model = EmbeddingModel()
    x = tf.ones([2, 128, 128, 1])
    embeddings, segmentation = model(x)
    assert embeddings.shape == (2, 128, 128, 8)
    assert segmentation.shape == (2, 128, 128, 2)
