from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from skimage.color import label2rgb
import typer

from instancesegmentation.data.circledata import example_data_generator


def main(export_path: Optional[Path] = None, seed: int = 1337) -> None:
    """Plot three random examples of the circle data set."""
    if export_path is None:
        export_path = Path(".validation_output")

    export_path.mkdir(parents=True, exist_ok=True)
    np.random.seed(seed)

    data_generator = example_data_generator()

    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(10, 15))
    for i in range(3):
        image, instance_label, _ = next(data_generator)
        axes[i][0].imshow(image, cmap="gray", vmin=0.0, vmax=1.0)
        axes[i][0].set_axis_off()
        axes[i][1].imshow(label2rgb(instance_label[..., 0], image[..., 0]))
        axes[i][1].set_axis_off()

    plt.savefig(export_path / "circle_data.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    typer.run(main)
