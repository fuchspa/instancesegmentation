from instancesegmentation.loss.metricloss import MetricLoss


def test_creating_the_loss_layer() -> None:
    metric_loss = MetricLoss()
    assert metric_loss.name == "MetricLoss"


def test_loss_layer_is_not_trainable() -> None:
    metric_loss = MetricLoss()
    assert metric_loss.trainable is False
