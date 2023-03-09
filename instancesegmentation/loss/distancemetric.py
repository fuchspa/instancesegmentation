from abc import ABC, abstractmethod

import tensorflow as tf


class DistanceMetric(ABC):
    """
    Computes a sample-wise distance between two tensors,
    assuming the features along the last dimension.
    """

    @abstractmethod
    def __call__(self, a: tf.Tensor, b: tf.Tensor) -> tf.Tensor:
        pass


class ManhattanDistanceMetric(DistanceMetric):
    """
    Compute a sample-wise Manhattan (L1) distance between to tensors,
    assuming the features along the last dimension.
    """

    def __call__(self, a: tf.Tensor, b: tf.Tensor) -> tf.Tensor:
        return tf.math.reduce_sum(tf.math.abs(a - b), axis=-1)


class EuclideanDistanceMetric(DistanceMetric):
    """
    Compute a sample-wise Euclidean (L2) distance between to tensors,
    assuming the features along the last dimension.
    """

    def __call__(self, a: tf.Tensor, b: tf.Tensor) -> tf.Tensor:
        return tf.math.sqrt(tf.math.reduce_sum(tf.math.square(a - b), axis=-1))
