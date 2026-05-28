from __future__ import annotations

import argparse
import json
import os
import random
import tarfile
import urllib.request
from pathlib import Path


DATASET_URL = "https://www.dropbox.com/s/zcwlujrtz3izcw8/gender.tgz?dl=1"
ARRAY_FILES = ("x_train.npy", "x_test.npy", "y_train.npy", "y_test.npy")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CNNs for gender recognition.")
    parser.add_argument("--model", choices=("small", "strong", "both"), default="both")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--out-dir", type=Path, default=Path("results_gender"))
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("."),
        help="Directory containing the gender .npy files. Defaults to the current directory.",
    )
    return parser.parse_args()


def set_seed(seed: int, tf) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass


def has_dataset_arrays(data_dir: Path) -> bool:
    return all((data_dir / filename).exists() for filename in ARRAY_FILES)


def safe_extract(tar: tarfile.TarFile, path: Path) -> None:
    path = path.resolve()
    for member in tar.getmembers():
        target = (path / member.name).resolve()
        try:
            target.relative_to(path)
        except ValueError:
            raise RuntimeError(f"Refusing to extract unsafe path: {member.name}")
    tar.extractall(path)


def download_and_extract(data_dir: Path) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    archive_path = data_dir / "gender.tgz"

    if not archive_path.exists():
        print(f"Downloading dataset to {archive_path} ...")
        urllib.request.urlretrieve(DATASET_URL, archive_path)
    else:
        print(f"Using existing archive: {archive_path}")

    print(f"Extracting {archive_path} ...")
    with tarfile.open(archive_path, "r:gz") as tar:
        safe_extract(tar, data_dir)


def resolve_data_dir(data_dir: Path) -> Path:
    data_dir = data_dir.expanduser()
    if has_dataset_arrays(data_dir):
        return data_dir

    notebook_dir = Path("notebook")
    if data_dir == Path(".") and has_dataset_arrays(notebook_dir):
        print("Dataset arrays not found in '.', using existing notebook/ arrays.")
        return notebook_dir

    download_and_extract(data_dir)
    if has_dataset_arrays(data_dir):
        return data_dir

    raise FileNotFoundError(
        "Could not find x_train.npy, x_test.npy, y_train.npy and y_test.npy "
        f"in {data_dir} after extraction."
    )


def labels_to_int(labels: np.ndarray, mapping: dict[int, int] | None = None) -> tuple[np.ndarray, dict[int, int]]:
    labels = np.asarray(labels)
    if labels.ndim > 1 and labels.shape[-1] > 1:
        labels = np.argmax(labels, axis=-1)
    labels = labels.reshape(-1)

    if mapping is None:
        unique = np.unique(labels)
        mapping = {int(value): index for index, value in enumerate(unique)}
    unknown = sorted(set(int(value) for value in np.unique(labels)) - set(mapping))
    if unknown:
        raise ValueError(f"Found labels in evaluation data that were not present in training: {unknown}")

    converted = np.array([mapping[int(value)] for value in labels], dtype=np.int64)
    return converted, mapping


def load_dataset(data_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[int, int]]:
    data_dir = resolve_data_dir(data_dir)
    print(f"Loading dataset from {data_dir}")

    x_train = np.load(data_dir / "x_train.npy").astype("float32") / 255.0
    x_test = np.load(data_dir / "x_test.npy").astype("float32") / 255.0
    y_train, label_mapping = labels_to_int(np.load(data_dir / "y_train.npy"))
    y_test, _ = labels_to_int(np.load(data_dir / "y_test.npy"), label_mapping)

    if x_train.ndim != 4 or x_test.ndim != 4:
        raise ValueError("Expected RGB image arrays with shape (N, H, W, C).")
    if x_train.shape[-1] != 3 or x_test.shape[-1] != 3:
        raise ValueError("Expected RGB images with 3 channels.")

    print(f"x_train: {x_train.shape}, y_train: {y_train.shape}")
    print(f"x_test:  {x_test.shape}, y_test:  {y_test.shape}")
    print_class_distribution("train", y_train)
    print_class_distribution("test", y_test)
    return x_train, y_train, x_test, y_test, label_mapping


def print_class_distribution(name: str, labels: np.ndarray) -> None:
    labels = np.asarray(labels)
    total = labels.size
    print(f"{name} class distribution:")
    for label in sorted(np.unique(labels)):
        count = int(np.sum(labels == label))
        print(f"  class {int(label)}: {count} ({count / total:.4f})")


def train_val_split(
    x_train: np.ndarray,
    y_train: np.ndarray,
    val_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train_indices = []
    val_indices = []

    for label in sorted(np.unique(y_train)):
        indices = np.flatnonzero(y_train == label)
        rng.shuffle(indices)
        val_count = max(1, int(round(indices.size * val_fraction)))
        val_indices.extend(indices[:val_count])
        train_indices.extend(indices[val_count:])

    train_indices = np.array(train_indices, dtype=np.int64)
    val_indices = np.array(val_indices, dtype=np.int64)
    rng.shuffle(train_indices)
    rng.shuffle(val_indices)

    return x_train[train_indices], y_train[train_indices], x_train[val_indices], y_train[val_indices]


def augmentation_layers(seed: int, tf):
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal", seed=seed),
            tf.keras.layers.RandomRotation(0.04, fill_mode="nearest", seed=seed + 1),
            tf.keras.layers.RandomTranslation(0.05, 0.05, fill_mode="nearest", seed=seed + 2),
            tf.keras.layers.RandomZoom(0.08, fill_mode="nearest", seed=seed + 3),
        ],
        name="light_augmentation",
    )


def conv_bn_relu(x, filters: int, tf, kernel_size: int = 3):
    x = tf.keras.layers.Conv2D(
        filters,
        kernel_size,
        padding="same",
        use_bias=False,
        kernel_initializer="he_normal",
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    return tf.keras.layers.Activation("relu")(x)


def separable_bn_relu(x, filters: int, tf):
    x = tf.keras.layers.SeparableConv2D(
        filters,
        3,
        padding="same",
        use_bias=False,
        depthwise_initializer="he_normal",
        pointwise_initializer="he_normal",
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    return tf.keras.layers.Activation("relu")(x)


def build_small_model(input_shape: tuple[int, int, int], num_classes: int, augment: bool, seed: int, tf):
    inputs = tf.keras.Input(shape=input_shape)
    x = augmentation_layers(seed, tf)(inputs) if augment else inputs

    x = conv_bn_relu(x, 24, tf)
    x = separable_bn_relu(x, 48, tf)
    x = separable_bn_relu(x, 48, tf)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Dropout(0.15)(x)

    x = separable_bn_relu(x, 96, tf)
    x = separable_bn_relu(x, 96, tf)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Dropout(0.20)(x)

    x = separable_bn_relu(x, 128, tf)
    x = separable_bn_relu(x, 128, tf)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.30)(x)
    x = tf.keras.layers.Dense(96, activation="relu", kernel_initializer="he_normal")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.35)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs, name="gender_small_cnn")
    trainable_params = count_trainable_params(model)
    if trainable_params >= 100000:
        raise ValueError(f"Small model has {trainable_params} trainable parameters, expected < 100000.")
    return model


def build_strong_model(input_shape: tuple[int, int, int], num_classes: int, augment: bool, seed: int, tf):
    inputs = tf.keras.Input(shape=input_shape)
    x = augmentation_layers(seed, tf)(inputs) if augment else inputs

    for filters, dropout in ((32, 0.10), (64, 0.15), (128, 0.25), (256, 0.35)):
        x = conv_bn_relu(x, filters, tf)
        x = conv_bn_relu(x, filters, tf)
        x = tf.keras.layers.MaxPooling2D()(x)
        x = tf.keras.layers.Dropout(dropout)(x)

    x = conv_bn_relu(x, 384, tf)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.40)(x)
    x = tf.keras.layers.Dense(256, activation="relu", kernel_initializer="he_normal")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.45)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    return tf.keras.Model(inputs, outputs, name="gender_strong_cnn")


def count_trainable_params(model) -> int:
    return int(sum(np.prod(weight.shape) for weight in model.trainable_weights))


def count_total_params(model) -> int:
    return int(model.count_params())


def compile_model(model, lr: float, tf) -> None:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )


def make_callbacks(name: str, out_dir: Path, tf):
    checkpoint_path = out_dir / f"best_{name}.keras"
    return [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            mode="max",
            patience=10,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=4,
            min_lr=1e-6,
            verbose=1,
        ),
    ]


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for true_label, pred_label in zip(y_true, y_pred):
        matrix[int(true_label), int(pred_label)] += 1
    return matrix


def classification_metrics(matrix: np.ndarray, class_names: list[str]) -> tuple[str, dict[str, dict[str, float]]]:
    lines = ["class precision recall f1-score support"]
    metrics = {}

    for index, class_name in enumerate(class_names):
        tp = float(matrix[index, index])
        fp = float(matrix[:, index].sum() - matrix[index, index])
        fn = float(matrix[index, :].sum() - matrix[index, index])
        support = float(matrix[index, :].sum())
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        metrics[class_name] = {
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "support": support,
        }
        lines.append(f"{class_name:>7} {precision:9.4f} {recall:6.4f} {f1:8.4f} {int(support):7d}")

    accuracy = float(np.trace(matrix) / matrix.sum()) if matrix.sum() else 0.0
    lines.append(f"\naccuracy {accuracy:.4f}")
    return "\n".join(lines), metrics


def plot_confusion_matrix(matrix: np.ndarray, class_names: list[str], path: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 4))
    image = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(len(class_names)), labels=class_names)
    ax.set_yticks(range(len(class_names)), labels=class_names)

    threshold = matrix.max() / 2 if matrix.size else 0
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            color = "white" if matrix[row, col] > threshold else "black"
            ax.text(col, row, str(matrix[row, col]), ha="center", va="center", color=color)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_history(history: dict[str, list[float]], path: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    epochs = range(1, len(history.get("loss", [])) + 1)

    axes[0].plot(epochs, history.get("loss", []), label="train")
    axes[0].plot(epochs, history.get("val_loss", []), label="validation")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(epochs, history.get("accuracy", []), label="train")
    axes[1].plot(epochs, history.get("val_accuracy", []), label="validation")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def to_jsonable_history(history: dict[str, list[float]]) -> dict[str, list[float]]:
    return {key: [float(value) for value in values] for key, values in history.items()}


def train_and_evaluate(
    name: str,
    build_model,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    args: argparse.Namespace,
    class_names: list[str],
    tf,
) -> dict[str, object]:
    print(f"\nTraining {name} model")
    model = build_model(x_train.shape[1:], len(class_names), args.augment, args.seed, tf)
    compile_model(model, args.lr, tf)
    model.summary()

    trainable_params = count_trainable_params(model)
    total_params = count_total_params(model)
    print(f"{name} trainable parameters: {trainable_params}")
    print(f"{name} total parameters: {total_params}")

    callbacks = make_callbacks(name, args.out_dir, tf)
    history_obj = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        shuffle=True,
        callbacks=callbacks,
        verbose=1,
    )
    history = to_jsonable_history(history_obj.history)

    checkpoint_path = args.out_dir / f"best_{name}.keras"
    if checkpoint_path.exists():
        model = tf.keras.models.load_model(checkpoint_path)

    test_loss, test_accuracy = model.evaluate(x_test, y_test, batch_size=args.batch_size, verbose=0)
    probabilities = model.predict(x_test, batch_size=args.batch_size, verbose=0)
    y_pred = np.argmax(probabilities, axis=1)

    matrix = confusion_matrix(y_test, y_pred, len(class_names))
    report_text, report_metrics = classification_metrics(matrix, class_names)

    report_path = args.out_dir / f"classification_report_{name}.txt"
    report_path.write_text(report_text + "\n", encoding="utf-8")
    plot_confusion_matrix(
        matrix,
        class_names,
        args.out_dir / f"confusion_matrix_{name}.png",
        f"{name} confusion matrix",
    )
    plot_history(history, args.out_dir / f"history_{name}.png", f"{name} training history")

    model_summary = {
        "test_loss": float(test_loss),
        "test_accuracy": float(test_accuracy),
        "trainable_params": trainable_params,
        "total_params": total_params,
        "best_model_path": str(checkpoint_path),
        "classification_report_path": str(report_path),
        "confusion_matrix": matrix.tolist(),
        "classification_report": report_metrics,
        "history": history,
    }
    print(f"{name} final test accuracy: {test_accuracy:.4f}")
    return model_summary


def write_summaries(out_dir: Path, summary: dict[str, object]) -> None:
    json_path = out_dir / "summary.json"
    txt_path = out_dir / "summary.txt"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = ["Gender recognition experiment summary", ""]
    for name, model_summary in summary["models"].items():
        lines.append(f"{name} model")
        lines.append(f"  test_accuracy: {model_summary['test_accuracy']:.4f}")
        lines.append(f"  test_loss: {model_summary['test_loss']:.4f}")
        lines.append(f"  trainable_params: {model_summary['trainable_params']}")
        lines.append(f"  total_params: {model_summary['total_params']}")
        lines.append(f"  best_model_path: {model_summary['best_model_path']}")
        lines.append("")
    txt_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    global np

    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    import numpy as np
    import tensorflow as tf

    set_seed(args.seed, tf)
    print(f"TensorFlow version: {tf.__version__}")
    print(f"GPUs available: {len(tf.config.list_physical_devices('GPU'))}")

    x_train_full, y_train_full, x_test, y_test, label_mapping = load_dataset(args.data_dir)
    x_train, y_train, x_val, y_val = train_val_split(x_train_full, y_train_full, 0.10, args.seed)
    print_class_distribution("validation", y_val)

    class_names = [
        f"label_{original_label}"
        for original_label, _ in sorted(label_mapping.items(), key=lambda item: item[1])
    ]
    models_to_run = ["small", "strong"] if args.model == "both" else [args.model]
    builders = {
        "small": build_small_model,
        "strong": build_strong_model,
    }

    summary = {
        "args": {
            "model": args.model,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "out_dir": str(args.out_dir),
            "augment": args.augment,
            "seed": args.seed,
            "data_dir": str(args.data_dir),
        },
        "dataset": {
            "train_shape": list(x_train_full.shape),
            "test_shape": list(x_test.shape),
            "label_mapping": {str(key): value for key, value in label_mapping.items()},
        },
        "models": {},
    }

    for model_name in models_to_run:
        summary["models"][model_name] = train_and_evaluate(
            model_name,
            builders[model_name],
            x_train,
            y_train,
            x_val,
            y_val,
            x_test,
            y_test,
            args,
            class_names,
            tf,
        )

    write_summaries(args.out_dir, summary)
    print(f"\nWrote summary files to {args.out_dir}")


if __name__ == "__main__":
    main()
