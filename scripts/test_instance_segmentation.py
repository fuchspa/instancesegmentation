from pathlib import Path
from typing import Optional

from hdbscan import HDBSCAN
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from skimage.color import label2rgb
import tensorflow as tf
import typer

from instancesegmentation.data.circledata import example_data_generator
from instancesegmentation.model.embeddingmodel import EmbeddingModel


def create_test_data() -> tuple[NDArray, NDArray, NDArray]:
    """
    Draw a sample from the data distribution.
    """
    generator = example_data_generator()
    return generator.__next__()


def plot_embeddings(output_path: Path, embeddings: NDArray) -> None:
    """
    Arrange the embeddings in a nice grid.
    """
    embedding_count = embeddings.shape[-1]
    fig, axes = plt.subplots(nrows=embedding_count // 4, ncols=4, figsize=(25, 15))
    for i in range(embedding_count):
        axes[i // 4][i % 4].imshow(embeddings[:, :, i])
        axes[i // 4][i % 4].set_axis_off()
    plt.savefig(output_path / "embeddings.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


def cluster_foreground_embeddings(
    embeddings: NDArray, segmentation: NDArray
) -> NDArray:
    """
    Cluster all the pixels belonging to the foreground class to separate instances.
    Background is labeled 0.
    """
    relevant_embeddings = embeddings[segmentation, :].reshape((-1, 8))
    relevant_clusters = HDBSCAN(metric="manhattan").fit_predict(relevant_embeddings)
    clusters = np.zeros(segmentation.shape, np.int32)
    clusters[segmentation] = relevant_clusters + 1
    return clusters


def plot_instance_segmentation(
    output_path: Path, image: NDArray, clusters: NDArray
) -> None:
    """
    Colorize the image data according to the assigned labels.
    """
    fig = plt.figure(figsize=(10, 10))
    plt.imshow(label2rgb(clusters, image[..., 0]))
    plt.savefig(output_path / "clustering.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


def main(
    model_path: Path, output_path: Optional[Path] = None, seed: int = 1337
) -> None:
    np.random.seed(seed)
    if output_path is None:
        output_path = Path(".validation_output")

    model = EmbeddingModel()
    checkpoint = tf.train.Checkpoint(net=model)
    checkpoint.restore(tf.train.latest_checkpoint(model_path)).expect_partial()

    image, _, _ = create_test_data()
    embeddings, segmentation = model(tf.constant(image[np.newaxis, ...], tf.float32))
    segmentation_numpy = tf.nn.softmax(segmentation)[0, ..., 1].numpy() > 0.5
    embeddings_numpy = tf.nn.sigmoid(embeddings)[0, ...].numpy()

    plot_embeddings(output_path, embeddings_numpy)

    clusters = cluster_foreground_embeddings(embeddings_numpy, segmentation_numpy)

    plot_instance_segmentation(output_path, image, clusters)


if __name__ == "__main__":
    typer.run(main)
