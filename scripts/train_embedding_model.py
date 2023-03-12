from pathlib import Path

import tensorflow as tf
import typer

from instancesegmentation.data.circledata import example_data_generator
from instancesegmentation.loss.metricloss import MetricLoss
from instancesegmentation.model.embeddingmodel import EmbeddingModel


def main(
    export_path: Path, report_path: Path, number_of_iterations: int = 5_000
) -> None:
    data_set = (
        tf.data.Dataset.from_generator(
            example_data_generator,
            output_types=(tf.float32, tf.int32, tf.float32),
            output_shapes=((512, 512, 1), (512, 512, 1), (512, 512, 1)),
        )
        .batch(4)
        .prefetch(4)
    )
    model = EmbeddingModel()
    embedding_loss = MetricLoss()
    segmentation_loss = tf.keras.losses.SparseCategoricalCrossentropy()

    learning_rate = tf.keras.optimizers.schedules.InverseTimeDecay(1e-3, 1, 1e-4)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    checkpoint = tf.train.Checkpoint(
        step=tf.Variable(0, name="step"), optimizer=optimizer, net=model
    )
    manager = tf.train.CheckpointManager(checkpoint, export_path, max_to_keep=None)
    file_writer = tf.summary.create_file_writer(str(report_path))

    with file_writer.as_default():
        for data, instance_labels, segmentation_labels in data_set:
            i = int(checkpoint.step)
            with tf.GradientTape() as tape:
                embeddings, segmentations = model(data)
                embeddings = tf.nn.sigmoid(embeddings)
                segmentations = tf.nn.softmax(segmentations)

                segmentation_loss_value = segmentation_loss(
                    segmentation_labels, segmentations
                )
                embedding_loss_value = embedding_loss(embeddings, instance_labels)
                regularization_loss_value = tf.reduce_mean(model.losses)
                total_loss = (
                    segmentation_loss_value
                    + embedding_loss_value
                    + 1e-4 * regularization_loss_value
                )

                grads = tape.gradient(total_loss, model.trainable_weights)
                optimizer.apply_gradients(zip(grads, model.trainable_weights))

            if i % 1 == 0:
                with tf.name_scope("Losses"):
                    tf.summary.scalar(
                        "Segmentation", data=segmentation_loss_value, step=i
                    )
                    tf.summary.scalar("Embedding", data=embedding_loss_value, step=i)
                    tf.summary.scalar(
                        "Regularization", data=regularization_loss_value, step=i
                    )
                with tf.name_scope("Training"):
                    tf.summary.scalar("Learning rate", data=learning_rate(i), step=i)
            if i % 10 == 0:
                with tf.name_scope("Training"):
                    tf.summary.image("Training data", data, step=i)
                with tf.name_scope("Embeddings"):
                    tf.summary.image("Embeddings 0", embeddings[..., 0:1], step=i)
                    tf.summary.image("Embeddings 1", embeddings[..., 1:2], step=i)
                    tf.summary.image("Embeddings 2", embeddings[..., 2:3], step=i)
                with tf.name_scope("Segmentation"):
                    tf.summary.image("Segmentation", segmentations[..., 1:2], step=i)
            if i % 1000 == 0:
                manager.save()
            if i > 0 and i % number_of_iterations == 0:
                break

            checkpoint.step.assign_add(1)


if __name__ == "__main__":
    typer.run(main)
