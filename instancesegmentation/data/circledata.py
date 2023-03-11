from dataclasses import dataclass
from typing import Iterator, Union

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


def normalize_data_by_dtype(image: NDArray) -> NDArray:
    """Convert the integer image to float and normalize with respect to the maximum datatype range."""
    return (image.astype(np.float32) - np.iinfo(image.dtype).min) / (
        np.iinfo(image.dtype).max - np.iinfo(image.dtype).min
    )


def add_white_noise(image: NDArray, percentage: float) -> NDArray:
    """Add a given amount of noise to the image."""
    return (1.0 - percentage) * image + percentage * np.random.standard_normal(
        image.shape
    )


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


def example_data_generator() -> Iterator[tuple[NDArray, NDArray, NDArray]]:
    """Generates a tuple of data, instance label, and segmentation label."""
    while True:
        number_of_circles = int(np.random.uniform(25, 75))

        image, label = create_data_point(
            Size(512, 512),
            number_of_circles,
            BackgroundProperties(gray_value=Range(4_000, 12_000)),
            CircleProperties(radius=Range(50, 135), gray_value=Range(35_000, 45_000)),
        )

        image = normalize_data_by_dtype(image)
        image = add_white_noise(image, 0.05)
        image = (image - image.min()) / (image.max() - image.min())

        image = np.expand_dims(image, axis=2)
        instance_label = np.expand_dims(label.astype(np.float32), axis=2)
        segmentation_label = np.expand_dims((label > 0).astype(np.float32), axis=2)

        yield image, instance_label, segmentation_label
