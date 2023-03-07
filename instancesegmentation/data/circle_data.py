from dataclasses import dataclass
from typing import Union

import numpy as np
from numpy.typing import DTypeLike, NDArray


@dataclass
class Size:
    x: int
    y: int


@dataclass
class Range:
    a: Union[int, float]
    b: Union[int, float]


@dataclass
class BackgroundProperties:
    gray_value: Range


@dataclass
class CircleProperties:
    radius: Range
    gray_value: Range


def initialize_empty_image(
    size: Size, dtype: DTypeLike, background: int = 0
) -> NDArray:
    """Returns an empty canvas."""
    return np.ones((size.x, size.y), dtype) * background


def select_a_random_position(extent: tuple[int, ...]) -> tuple[int, ...]:
    """Randomly select a point within a given extent."""
    return tuple([int(np.random.uniform(low=0, high=e)) for e in extent])


def create_data_point(
    size: Size,
    n: int,
    background_properties: BackgroundProperties,
    circle_properties: CircleProperties,
) -> tuple[NDArray, NDArray]:
    """Creates an imaga with n spheres and the according instance segmentation ground truth."""
    if n >= np.iinfo(np.uint8).max:
        raise Exception("Can only create up to 255 circles.")

    background = int(
        np.random.uniform(
            background_properties.gray_value.a, background_properties.gray_value.b
        )
    )
    image = initialize_empty_image(size, np.uint16, background=background)
    label = initialize_empty_image(size, np.uint8)

    xx, yy = np.meshgrid(range(size.x), range(size.y))

    for i in range(n):
        pos = select_a_random_position(image.shape)
        radius = np.random.uniform(
            circle_properties.radius.a, circle_properties.radius.b
        )
        gray_value = np.random.uniform(
            circle_properties.gray_value.a, circle_properties.gray_value.b
        )
        circle = (xx - pos[0]) ** 2 + (yy - pos[1]) ** 2 < radius
        image[circle] = gray_value
        label[circle] = i + 1

    return image, label
