import numpy as np
import pytest

from instancesegmentation.data.circledata import (
    BackgroundProperties,
    CircleProperties,
    create_data_point,
    initialize_empty_image,
    Range,
    select_a_random_position,
    Size,
)


def test_initialize_empty_image() -> None:
    desired_size = Size(512, 512)
    desired_dtype = np.uint16
    image = initialize_empty_image(desired_size, desired_dtype)

    assert image.shape[0] == desired_size.x and image.shape[1] == desired_size.y
    assert image.dtype == desired_dtype
    assert np.all(image == 0)


def test_initialize_label_image() -> None:
    desired_size = Size(128, 77)
    desired_dtype = np.uint8
    image = initialize_empty_image(desired_size, desired_dtype)

    assert image.shape[0] == desired_size.x and image.shape[1] == desired_size.y
    assert image.dtype == desired_dtype
    assert np.all(image == 0)


def test_initialize_image_with_background() -> None:
    desired_size = Size(512, 512)
    desired_dtype = np.uint16
    image = initialize_empty_image(desired_size, desired_dtype, 4_500)

    assert image.shape[0] == desired_size.x and image.shape[1] == desired_size.y
    assert image.dtype == desired_dtype
    assert np.all(image == 4_500)


@pytest.mark.parametrize("extent", [(20, 20), (15, 20, 13, 1), (4, 23, 1), (100,)])
def test_select_random_position(extent: tuple[int, ...]) -> None:
    np.random.seed(1337)

    pos = select_a_random_position(extent)
    assert len(pos) == len(extent)
    assert np.all(np.array(pos) < np.array(extent))


def test_create_data_point() -> None:
    for _ in range(10):
        number_of_circles = int(np.random.uniform(30, 60))
        image, label = create_data_point(
            Size(512, 512),
            number_of_circles,
            background_properties=BackgroundProperties(Range(4_000, 12_000)),
            circle_properties=CircleProperties(
                radius=Range(50, 65), gray_value=Range(35_000, 45_000)
            ),
        )
        assert label.max() == number_of_circles
        for index in range(number_of_circles):
            values = image[label == index + 1]
            assert np.all(values == values[0])
