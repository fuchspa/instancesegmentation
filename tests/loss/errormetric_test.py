from math import cosh, log

import tensorflow as tf

from instancesegmentation.loss.errormetric import (
    LogCoshErrorMetric,
    SquaredErrorMetric,
)


def test_squared_error() -> None:
    error_metric = SquaredErrorMetric()
    distance = tf.constant([0.5, 1.0, 2.0, 4.0])

    loss = error_metric(distance)

    expected_loss = tf.constant([0.25, 1.0, 4.0, 16.0])
    assert tf.reduce_all(tf.math.abs(loss - expected_loss) < 1e-7)


def test_logcosh_error() -> None:
    error_metric = LogCoshErrorMetric()
    distance = tf.constant([0.5, 1.0, 2.0, 4.0])

    loss = error_metric(distance)

    expected_loss = tf.constant(
        [log(cosh(0.5)), log(cosh(1.0)), log(cosh(2.0)), log(cosh(4.0))]
    )
    assert tf.reduce_all(tf.math.abs(loss - expected_loss) < 1e-7)
