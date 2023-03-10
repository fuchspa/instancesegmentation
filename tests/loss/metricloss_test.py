import tensorflow as tf

from instancesegmentation.loss.metricloss import (
    compute_centroids,
    MetricLoss,
    strictly_upper_triangular_matrix,
)


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

    centroids, instance_sizes = compute_centroids(embeddings, labels)

    expected_centroids = tf.constant([[0.5, 0.5], [0.2, 0.2], [0.4, 0.4]])
    expected_instance_sizes = tf.constant([2, 3, 1], tf.int32)
    assert tf.reduce_all(centroids == expected_centroids)
    assert tf.reduce_all(instance_sizes == expected_instance_sizes)


def test_compute_centroids_can_deal_with_missing_labels() -> None:
    embeddings = tf.constant(
        [[1.0, 1.0], [0.0, 0.0], [0.1, 0.3], [0.3, 0.1]],
        dtype=tf.float32,
    )
    labels = tf.constant([0, 0, 3, 3], tf.int32)

    centroids, instance_sizes = compute_centroids(embeddings, labels)

    expected_centroids = tf.constant([[0.5, 0.5], [0.0, 0.0], [0.0, 0.0], [0.2, 0.2]])
    expected_instance_sizes = tf.constant([2, 0, 0, 2], tf.int32)
    assert tf.reduce_all(centroids == expected_centroids)
    assert tf.reduce_all(instance_sizes == expected_instance_sizes)


def test_strictly_upper_triangular_matrix() -> None:
    matrix = tf.ones((3, 3))

    triu = strictly_upper_triangular_matrix(matrix)

    expected_matrix = tf.constant([[0.0, 1.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]])
    assert triu.dtype == matrix.dtype
    assert tf.reduce_all(triu == expected_matrix)


def test_compute_push_loss() -> None:
    push_margin = 0.25
    metric_loss = MetricLoss(push_margin=push_margin)
    centroids = tf.constant([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])

    loss = metric_loss._compute_push_loss(centroids)
    assert tf.math.abs(loss - push_margin * push_margin) < 1e-7

    centroids = tf.constant([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])

    loss = metric_loss._compute_push_loss(centroids)
    assert tf.math.abs(loss) < 1e-7

    centroids = tf.constant([[0.0, 0.0], [0.25, 0.0], [0.1, 0.0]])

    loss = metric_loss._compute_push_loss(centroids)
    assert tf.math.abs(loss) - 0.010833333 < 1e-7

    centroids = tf.constant([[5.0, 5.0]])

    loss = metric_loss._compute_push_loss(centroids)
    assert tf.math.abs(loss) < 1e-7


def test_compute_pull_loss() -> None:
    pull_margin = 0.1
    metric_loss = MetricLoss(pull_margin=pull_margin)
    embeddings = tf.constant([[0.0, 0.0], [2.0, 0.0], [0.0, 0.0], [0.0, 2.0]])
    centroids = tf.constant([[1.0, 0.0], [0.0, 0.0], [0.0, 1.0]])
    labels = tf.constant([0, 0, 2, 2])

    loss = metric_loss._compute_pull_loss(embeddings, centroids, labels)
    assert tf.math.abs(loss - 0.81) < 1e-7

    embeddings = tf.constant([[0.0, 0.0], [0.2, 0.0], [0.0, 0.0], [0.0, 0.2]])
    centroids = tf.constant([[0.1, 0.0], [0.0, 0.1]])
    labels = tf.constant([0, 0, 1, 1])

    loss = metric_loss._compute_pull_loss(embeddings, centroids, labels)
    assert tf.math.abs(loss) < 1e-7
