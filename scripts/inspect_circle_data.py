from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from skimage.color import label2rgb
import typer

from instancesegmentation.data.circle_data import (
    BackgroundProperties,
    CircleProperties,
    create_data_point,
    Range,
    Size,
)


def main(export_path: Optional[Path] = None, seed: int = 1337) -> None:
    """Plot three random examples of the circle data set."""
    if export_path is None:
        export_path = Path(".validation_output")

    export_path.mkdir(parents=True, exist_ok=True)
    np.random.seed(seed)

    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(10, 15))
    for i in range(3):
        circle_count = int(np.random.uniform(35, 60))
        image, label = create_data_point(
            Size(512, 512),
            circle_count,
            BackgroundProperties(gray_value=Range(4_000, 12_000)),
            CircleProperties(radius=Range(40, 85), gray_value=Range(35_000, 42_000)),
        )
        axes[i][0].imshow(
            image,
            cmap="gray",
            vmin=np.iinfo(image.dtype).min,
            vmax=np.iinfo(image.dtype).max,
        )
        axes[i][0].set_axis_off()
        axes[i][1].imshow(label2rgb(label, image))
        axes[i][1].set_axis_off()

    plt.savefig(export_path / "circle_data.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    typer.run(main)
