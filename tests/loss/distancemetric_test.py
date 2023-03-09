from math import sqrt

import tensorflow as tf

from instancesegmentation.loss.distancemetric import (
    EuclideanDistanceMetric,
    ManhattanDistanceMetric,
)


def test_manhattan_distance() -> None:
    distance_measure = ManhattanDistanceMetric()
    a = tf.constant([[1.0, 0.0], [0.0, -1.0]])
    b = tf.constant([[0.0, 1.0], [-1.0, 0.0]])

    distance = distance_measure(a, b)

    expected_distance = tf.constant([2.0, 2.0])
    assert tf.reduce_all(tf.math.abs(distance - expected_distance) < 1e-7)


def test_euclidean_distance() -> None:
    distance_measure = EuclideanDistanceMetric()
    a = tf.constant([[1.0, 0.0], [0.0, -1.0]])
    b = tf.constant([[0.0, 1.0], [-1.0, 0.0]])

    distance = distance_measure(a, b)

    expected_distance = tf.constant([sqrt(2.0), sqrt(2.0)])
    assert tf.reduce_all(tf.math.abs(distance - expected_distance) < 1e-7)
