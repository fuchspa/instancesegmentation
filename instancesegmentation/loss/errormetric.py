from abc import ABC, abstractmethod

import tensorflow as tf


class ErrorMetric(ABC):
    """Convert a difference to an error measure with a nice derivative."""

    @abstractmethod
    def __call__(self, values: tf.Tensor) -> tf.Tensor:
        pass


class SquaredErrorMetric(ErrorMetric):
    """Square the error to get a MSE-like loss."""

    def __call__(self, values: tf.Tensor) -> tf.Tensor:
        return tf.math.square(values)


class LogCoshErrorMetric(ErrorMetric):
    """
    Compute the log-cosh-value to get a more or less constant gradient
    for |values| > 1 and a MSE-like gradient for |values| < 1
    """

    def __call__(self, values: tf.Tensor) -> tf.Tensor:
        return tf.math.log(tf.math.cosh(values))
