#!/usr/bin/env python3
"""Train a simple Wide ResNet on CIFAR10 with TensorFlow/Keras."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers


CIFAR10_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a Wide ResNet on CIFAR10 and save metrics/plots."
    )
    parser.add_argument("--depth", type=int, default=16, help="WRN depth, depth = 6N + 4.")
    parser.add_argument("--width", type=int, default=4, help="Widening factor k.")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout inside residual blocks.")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=128, help="Training batch size.")
    parser.add_argument("--lr", type=float, default=0.1, help="Initial learning rate.")
    parser.add_argument("--out-dir", type=Path, default=Path("results_cifar_wrn"))
    parser.add_argument("--augment", action="store_true", help="Use random crop and horizontal flip.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--val-size",
        type=int,
        default=5000,
        help="Number of training images reserved for validation.",
    )
    return parser.parse_args()


def configure_runtime(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)

    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass

    for gpu in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass


def validate_wrn_args(depth: int, width: int, dropout: float) -> int:
    if depth < 10 or (depth - 4) % 6 != 0:
        raise ValueError("Wide ResNet depth must satisfy depth = 6N + 4, for example 16, 22 or 28.")
    if width < 1:
        raise ValueError("--width must be >= 1.")
    if not 0.0 <= dropout < 1.0:
        raise ValueError("--dropout must be in [0, 1).")
    return (depth - 4) // 6


def load_cifar10(val_size: int, seed: int):
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    y_train = y_train.reshape(-1).astype("int64")
    y_test = y_test.reshape(-1).astype("int64")

    if val_size <= 0:
        return (x_train, y_train), None, (x_test, y_test)
    if val_size >= len(x_train):
        raise ValueError("--val-size must be smaller than the CIFAR10 training set.")

    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(x_train))
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    train_data = (x_train[train_indices], y_train[train_indices])
    val_data = (x_train[val_indices], y_train[val_indices])
    test_data = (x_test, y_test)
    return train_data, val_data, test_data


def conv_bn_relu(x, filters: int, strides: int = 1, name: str | None = None):
    x = layers.Conv2D(
        filters,
        kernel_size=3,
        strides=strides,
        padding="same",
        use_bias=False,
        kernel_initializer="he_normal",
        kernel_regularizer=regularizers.l2(5e-4),
        name=None if name is None else f"{name}_conv",
    )(x)
    x = layers.BatchNormalization(name=None if name is None else f"{name}_bn")(x)
    x = layers.Activation("relu", name=None if name is None else f"{name}_relu")(x)
    return x


def residual_wide_block(
    inputs,
    filters: int,
    strides: int = 1,
    dropout: float = 0.0,
    name: str = "wrn_block",
):
    """A two-convolution Wide ResNet block with an optional projection shortcut."""
    x = layers.Conv2D(
        filters,
        kernel_size=3,
        strides=strides,
        padding="same",
        use_bias=False,
        kernel_initializer="he_normal",
        kernel_regularizer=regularizers.l2(5e-4),
        name=f"{name}_conv1",
    )(inputs)
    x = layers.BatchNormalization(name=f"{name}_bn1")(x)
    x = layers.Activation("relu", name=f"{name}_relu1")(x)

    if dropout > 0.0:
        x = layers.Dropout(dropout, name=f"{name}_dropout")(x)

    x = layers.Conv2D(
        filters,
        kernel_size=3,
        strides=1,
        padding="same",
        use_bias=False,
        kernel_initializer="he_normal",
        kernel_regularizer=regularizers.l2(5e-4),
        name=f"{name}_conv2",
    )(x)
    x = layers.BatchNormalization(name=f"{name}_bn2")(x)

    shortcut = inputs
    input_channels = int(inputs.shape[-1])
    if strides != 1 or input_channels != filters:
        shortcut = layers.Conv2D(
            filters,
            kernel_size=1,
            strides=strides,
            padding="same",
            use_bias=False,
            kernel_initializer="he_normal",
            kernel_regularizer=regularizers.l2(5e-4),
            name=f"{name}_projection",
        )(shortcut)
        shortcut = layers.BatchNormalization(name=f"{name}_projection_bn")(shortcut)

    x = layers.Add(name=f"{name}_add")([shortcut, x])
    x = layers.Activation("relu", name=f"{name}_out")(x)
    return x


def build_wide_resnet(
    input_shape=(32, 32, 3),
    depth: int = 16,
    width: int = 4,
    dropout: float = 0.0,
    num_classes: int = 10,
    augment: bool = False,
    seed: int = 42,
) -> keras.Model:
    num_blocks = validate_wrn_args(depth, width, dropout)
    inputs = keras.Input(shape=input_shape, name="image")

    x = inputs
    if augment:
        x = keras.Sequential(
            [
                layers.ZeroPadding2D(padding=4),
                layers.RandomCrop(height=32, width=32, seed=seed),
                layers.RandomFlip("horizontal", seed=seed + 1),
            ],
            name="augmentation",
        )(x)

    x = conv_bn_relu(x, 16, name="stem")

    for stage, filters in enumerate([16 * width, 32 * width, 64 * width]):
        for block in range(num_blocks):
            strides = 2 if stage > 0 and block == 0 else 1
            x = residual_wide_block(
                x,
                filters=filters,
                strides=strides,
                dropout=dropout,
                name=f"stage{stage + 1}_block{block + 1}",
            )

    x = layers.GlobalAveragePooling2D(name="global_average_pooling")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="classifier")(x)
    return keras.Model(inputs=inputs, outputs=outputs, name=f"WRN-{depth}-{width}")


def lr_schedule(initial_lr: float, total_epochs: int):
    def schedule(epoch: int, current_lr: float) -> float:
        if epoch >= int(total_epochs * 0.75):
            return initial_lr * 0.01
        if epoch >= int(total_epochs * 0.50):
            return initial_lr * 0.1
        return initial_lr

    return schedule


def plot_training_curves(history: dict, out_path: Path) -> None:
    epochs = range(1, len(history.get("loss", [])) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].plot(epochs, history.get("loss", []), label="train")
    if "val_loss" in history:
        axes[0].plot(epochs, history["val_loss"], label="validation")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Sparse categorical crossentropy")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(epochs, history.get("accuracy", []), label="train")
    if "val_accuracy" in history:
        axes[1].plot(epochs, history["val_accuracy"], label="validation")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for true_label, pred_label in zip(y_true, y_pred):
        matrix[int(true_label), int(pred_label)] += 1
    return matrix


def save_confusion_matrix(matrix: np.ndarray, out_dir: Path) -> None:
    csv_path = out_dir / "confusion_matrix.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["true\\pred", *CIFAR10_CLASSES])
        for class_name, row in zip(CIFAR10_CLASSES, matrix):
            writer.writerow([class_name, *row.tolist()])

    fig, ax = plt.subplots(figsize=(8, 7))
    image = ax.imshow(matrix, interpolation="nearest", cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set(
        xticks=np.arange(len(CIFAR10_CLASSES)),
        yticks=np.arange(len(CIFAR10_CLASSES)),
        xticklabels=CIFAR10_CLASSES,
        yticklabels=CIFAR10_CLASSES,
        xlabel="Predicted label",
        ylabel="True label",
        title="CIFAR10 confusion matrix",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    threshold = matrix.max() / 2.0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            color = "white" if matrix[i, j] > threshold else "black"
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center", color=color, fontsize=8)

    fig.tight_layout()
    fig.savefig(out_dir / "confusion_matrix.png", dpi=150)
    plt.close(fig)


def save_text_report(summary: dict, out_path: Path) -> None:
    lines = [
        "Wide ResNet on CIFAR10",
        "=======================",
        "",
        f"Model: WRN-{summary['depth']}-{summary['width']}",
        f"Residual blocks per stage N: {summary['blocks_per_stage']}",
        f"Dropout: {summary['dropout']}",
        f"Data augmentation: {summary['augment']}",
        f"Train images: {summary['train_size']}",
        f"Validation images: {summary['validation_size']}",
        f"Test images: {summary['test_size']}",
        f"Parameter count: {summary['parameter_count']}",
        f"Best validation accuracy: {summary['best_validation_accuracy']:.4f}",
        f"Test loss: {summary['test_loss']:.4f}",
        f"Test accuracy: {summary['test_accuracy']:.4f}",
        "",
        "Generated files:",
        "- best_wrn.keras",
        "- model_summary.txt",
        "- history.json",
        "- training_log.csv",
        "- training_curves.png",
        "- confusion_matrix.csv",
        "- confusion_matrix.png",
        "- summary.json",
        "- summary.txt",
    ]
    out_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    num_blocks = validate_wrn_args(args.depth, args.width, args.dropout)
    configure_runtime(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    train_data, val_data, test_data = load_cifar10(args.val_size, args.seed)
    x_train, y_train = train_data
    x_test, y_test = test_data

    validation_data = val_data if val_data is not None else test_data
    x_val, y_val = validation_data

    model = build_wide_resnet(
        depth=args.depth,
        width=args.width,
        dropout=args.dropout,
        augment=args.augment,
        seed=args.seed,
    )
    model.compile(
        optimizer=keras.optimizers.SGD(
            learning_rate=args.lr,
            momentum=0.9,
            nesterov=True,
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    summary_lines = []
    model.summary(print_fn=summary_lines.append)
    (args.out_dir / "model_summary.txt").write_text("\n".join(summary_lines) + "\n")

    best_model_path = args.out_dir / "best_wrn.keras"
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=str(best_model_path),
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        keras.callbacks.LearningRateScheduler(lr_schedule(args.lr, args.epochs), verbose=1),
        keras.callbacks.CSVLogger(str(args.out_dir / "training_log.csv")),
    ]

    history_obj = model.fit(
        x_train,
        y_train,
        batch_size=args.batch_size,
        epochs=args.epochs,
        validation_data=(x_val, y_val),
        shuffle=True,
        callbacks=callbacks,
        verbose=2,
    )

    history = {key: [float(value) for value in values] for key, values in history_obj.history.items()}
    (args.out_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    plot_training_curves(history, args.out_dir / "training_curves.png")

    if not best_model_path.exists():
        model.save(str(best_model_path))

    best_model = keras.models.load_model(str(best_model_path))
    test_loss, test_accuracy = best_model.evaluate(
        x_test,
        y_test,
        batch_size=args.batch_size,
        verbose=1,
    )
    probabilities = best_model.predict(x_test, batch_size=args.batch_size, verbose=0)
    predictions = np.argmax(probabilities, axis=1)
    matrix = confusion_matrix(y_test, predictions, len(CIFAR10_CLASSES))
    save_confusion_matrix(matrix, args.out_dir)

    best_val_accuracy = max(history.get("val_accuracy", [float("nan")]))
    summary = {
        "depth": args.depth,
        "width": args.width,
        "blocks_per_stage": num_blocks,
        "dropout": args.dropout,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "initial_learning_rate": args.lr,
        "augment": args.augment,
        "seed": args.seed,
        "train_size": int(len(x_train)),
        "validation_size": int(len(x_val)),
        "test_size": int(len(x_test)),
        "parameter_count": int(best_model.count_params()),
        "best_validation_accuracy": float(best_val_accuracy),
        "test_loss": float(test_loss),
        "test_accuracy": float(test_accuracy),
        "device": "GPU" if tf.config.list_logical_devices("GPU") else "CPU",
        "output_dir": str(args.out_dir),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    save_text_report(summary, args.out_dir / "summary.txt")

    print(f"Saved results to {args.out_dir}")
    print(f"Test accuracy: {test_accuracy:.4f}")
    print(f"Parameters: {best_model.count_params()}")


if __name__ == "__main__":
    main()
