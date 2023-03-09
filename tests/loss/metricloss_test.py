import numpy as np
import tensorflow as tf

from instancesegmentation.loss.metricloss import compute_centroids, MetricLoss


def test_creating_the_loss_layer() -> None:
    metric_loss = MetricLoss()
    assert metric_loss.name == "MetricLoss"


def test_loss_layer_is_not_trainable() -> None:
    metric_loss = MetricLoss()
    assert metric_loss.trainable is False


def test_compute_centroids() -> None:
    embeddings = tf.constant(
        [[1.0, 1.0], [0.0, 0.0], [0.2, 0.2], [0.1, 0.3], [0.3, 0.1], [0.4, 0.4]],
        dtype=tf.float32,
    )
    labels = tf.constant([0, 0, 1, 1, 1, 2], tf.int32)

    centroids = compute_centroids(embeddings, labels).numpy()

    expected_centroids = np.array([[0.5, 0.5], [0.2, 0.2], [0.4, 0.4]], np.float32)
    assert np.all(centroids == expected_centroids)


def test_compute_centroids_can_deal_with_missing_labels() -> None:
    embeddings = tf.constant(
        [[1.0, 1.0], [0.0, 0.0], [0.1, 0.3], [0.3, 0.1]],
        dtype=tf.float32,
    )
    labels = tf.constant([0, 0, 3, 3], tf.int32)

    centroids = compute_centroids(embeddings, labels).numpy()

    expected_centroids = np.array([[0.5, 0.5], [0.2, 0.2]], np.float32)
    assert np.all(centroids == expected_centroids)
