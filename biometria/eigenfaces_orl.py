import argparse
import csv
import os
import re
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None
    from PIL import Image


IMAGE_EXTENSIONS = {".pgm", ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
DEFAULT_DIMS = "1,2,3,5,10,15,20,30,40,60,80,100"
EPS = 1e-10


@dataclass(frozen=True)
class ORLSample:
    image: np.ndarray
    label: int
    image_number: int
    path: Path


def parse_resize(value: str | None) -> tuple[int, int] | None:
    if value is None or value == "0":
        return None

    match = re.fullmatch(r"(\d+)x(\d+)", value.lower())
    if match is None:
        raise ValueError("--resize debe ser 0 o tener formato anchoxalto, por ejemplo 92x112.")

    width = int(match.group(1))
    height = int(match.group(2))
    if width <= 0 or height <= 0:
        raise ValueError("--resize debe usar dimensiones positivas.")
    return width, height


def parse_dims(value: str) -> list[int]:
    dims = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not dims:
        raise ValueError("--dims debe contener al menos una dimension.")
    if any(k <= 0 for k in dims):
        raise ValueError("Todas las dimensiones de --dims deben ser enteros positivos.")
    return sorted(dict.fromkeys(dims))


def _read_grayscale(path: Path, resize: tuple[int, int] | None) -> np.ndarray:
    if cv2 is not None:
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"No se pudo leer la imagen: {path}")
        if resize is not None:
            image = cv2.resize(image, resize, interpolation=cv2.INTER_AREA)
    else:
        with Image.open(path) as img:
            img = img.convert("L")
            if resize is not None:
                img = img.resize(resize, Image.Resampling.BILINEAR)
            image = np.asarray(img)

    return image.astype(np.float32) / 255.0


def _write_grayscale(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image_u8 = np.clip(image, 0, 255).astype(np.uint8)
    if cv2 is not None:
        cv2.imwrite(str(path), image_u8)
    else:
        Image.fromarray(image_u8).save(path)


def to_uint8_visualization(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    min_value = float(np.min(image))
    max_value = float(np.max(image))
    if max_value - min_value < EPS:
        return np.zeros(image.shape, dtype=np.uint8)
    scaled = (image - min_value) * (255.0 / (max_value - min_value))
    return np.clip(scaled, 0, 255).astype(np.uint8)


def _subject_label(name: str) -> int | None:
    match = re.fullmatch(r"s?(\d+)", name.lower())
    return int(match.group(1)) if match else None


def _image_number(path: Path) -> int | None:
    return int(path.stem) if path.stem.isdigit() else None


def load_orl_dataset(root_dir: str | Path, resize: tuple[int, int] | None = None) -> tuple[list[ORLSample], np.ndarray]:
    root_dir = Path(root_dir)
    if not root_dir.exists():
        raise FileNotFoundError(f"No existe el directorio ORL: {root_dir}")

    samples: list[ORLSample] = []
    subject_dirs = []
    for child in root_dir.iterdir():
        if child.is_dir():
            label = _subject_label(child.name)
            if label is not None:
                subject_dirs.append((label, child))

    for label, subject_dir in sorted(subject_dirs, key=lambda item: item[0]):
        image_paths = []
        for path in subject_dir.iterdir():
            number = _image_number(path)
            if number is not None and path.suffix.lower() in IMAGE_EXTENSIONS:
                image_paths.append((number, path))

        for number, path in sorted(image_paths, key=lambda item: item[0]):
            image = _read_grayscale(path, resize=resize)
            samples.append(ORLSample(image=image, label=label, image_number=number, path=path))

    if not samples:
        raise ValueError(
            "No se encontraron imagenes ORL. Estructuras esperadas: orl/s1/1.pgm o orl/1/1.pgm."
        )

    first_shape = samples[0].image.shape
    for sample in samples:
        if sample.image.shape != first_shape:
            raise ValueError(
                "Todas las imagenes deben tener el mismo tamano. Usa --resize si la base no es homogenea."
            )

    labels = np.array([sample.label for sample in samples], dtype=np.int32)
    return samples, labels


def split_orl_train_test(
    images: list[ORLSample],
    labels: np.ndarray,
) -> tuple[list[ORLSample], np.ndarray, list[ORLSample], np.ndarray]:
    if len(images) != len(labels):
        raise ValueError("images y labels deben tener la misma longitud.")

    train_samples: list[ORLSample] = []
    test_samples: list[ORLSample] = []
    train_labels: list[int] = []
    test_labels: list[int] = []

    for sample, label in zip(images, labels):
        if 1 <= sample.image_number <= 5:
            train_samples.append(sample)
            train_labels.append(int(label))
        elif 6 <= sample.image_number <= 10:
            test_samples.append(sample)
            test_labels.append(int(label))

    if not train_samples or not test_samples:
        raise ValueError("El split ORL requiere imagenes numeradas 1-5 para train y 6-10 para test.")

    return (
        train_samples,
        np.array(train_labels, dtype=np.int32),
        test_samples,
        np.array(test_labels, dtype=np.int32),
    )


def _as_image_array(item: ORLSample | np.ndarray) -> np.ndarray:
    return item.image if isinstance(item, ORLSample) else item


def vectorize_images(images: list[ORLSample] | list[np.ndarray]) -> np.ndarray:
    vectors = []
    first_shape = None
    for item in images:
        image = _as_image_array(item).astype(np.float32, copy=False)
        if first_shape is None:
            first_shape = image.shape
        elif image.shape != first_shape:
            raise ValueError("Todas las imagenes deben tener el mismo tamano para vectorizar.")
        vectors.append(image.reshape(-1))

    if not vectors:
        raise ValueError("No hay imagenes para vectorizar.")
    return np.vstack(vectors).astype(np.float32)


def fit_eigenfaces(X_train: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X_train = np.asarray(X_train, dtype=np.float32)
    if X_train.ndim != 2:
        raise ValueError("X_train debe tener forma (n_imagenes, n_pixeles).")

    n, d = X_train.shape
    if n < 2:
        raise ValueError("Se necesitan al menos dos imagenes de entrenamiento.")

    mean_face = np.mean(X_train, axis=0, dtype=np.float64).astype(np.float32)
    A = X_train - mean_face

    # Matriz pequena para no diagonalizar una matriz de pixeles x pixeles.
    C = (A @ A.T) / float(d)
    eigvals_small, eigvecs_small = np.linalg.eigh(C.astype(np.float64))

    order = np.argsort(eigvals_small)[::-1]
    eigvals_small = eigvals_small[order]
    eigvecs_small = eigvecs_small[:, order]

    eigvals = (float(d) / float(n)) * eigvals_small
    tolerance = max(float(eigvals[0]) * 1e-7, EPS) if eigvals.size else EPS
    valid = eigvals > tolerance
    eigvals = eigvals[valid]
    eigvecs_small = eigvecs_small[:, valid]

    if eigvecs_small.size == 0:
        raise ValueError("No se pudieron obtener eigenvectores numericamente estables.")

    eigenvectors = A.T @ eigvecs_small
    norms = np.linalg.norm(eigenvectors, axis=0)
    stable = norms > EPS
    eigenvectors = eigenvectors[:, stable] / norms[stable]
    eigvals = eigvals[stable]

    # Asi no cambian de signo cada vez que se ejecuta.
    for col in range(eigenvectors.shape[1]):
        pivot = int(np.argmax(np.abs(eigenvectors[:, col])))
        if eigenvectors[pivot, col] < 0:
            eigenvectors[:, col] *= -1.0

    return mean_face.astype(np.float32), eigenvectors.astype(np.float32), eigvals.astype(np.float32)


def project(X: np.ndarray, mean_face: np.ndarray, eigenvectors: np.ndarray, k: int) -> np.ndarray:
    if k < 1:
        raise ValueError("k debe ser positivo.")
    if k > eigenvectors.shape[1]:
        raise ValueError(f"k={k} supera el numero de eigenfaces disponibles ({eigenvectors.shape[1]}).")
    return (np.asarray(X, dtype=np.float32) - mean_face) @ eigenvectors[:, :k]


def nearest_neighbor_predict(
    Z_train: np.ndarray,
    y_train: np.ndarray,
    Z_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    train_norms = np.sum(Z_train * Z_train, axis=1)[None, :]
    test_norms = np.sum(Z_test * Z_test, axis=1)[:, None]
    distances_sq = np.maximum(test_norms + train_norms - 2.0 * (Z_test @ Z_train.T), 0.0)
    nearest = np.argmin(distances_sq, axis=1)
    predictions = y_train[nearest]
    distances = np.sqrt(distances_sq[np.arange(Z_test.shape[0]), nearest])
    return predictions.astype(np.int32), distances.astype(np.float32)


def evaluate_for_dimensions(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    mean_face: np.ndarray,
    eigenvectors: np.ndarray,
    dims: list[int],
    test_paths: list[str] | None = None,
) -> tuple[list[dict[str, int | float]], list[dict[str, int | float | str | bool]]]:
    results: list[dict[str, int | float]] = []
    prediction_rows: list[dict[str, int | float | str | bool]] = []
    max_k = eigenvectors.shape[1]

    for k in dims:
        if k > max_k:
            continue

        Z_train = project(X_train, mean_face, eigenvectors, k)
        Z_test = project(X_test, mean_face, eigenvectors, k)
        y_pred, distances = nearest_neighbor_predict(Z_train, y_train, Z_test)
        correct = y_pred == y_test
        num_errors = int(np.sum(~correct))
        num_test = int(len(y_test))
        accuracy = float(np.mean(correct))
        error_rate = float(num_errors / num_test)

        results.append(
            {
                "k": int(k),
                "accuracy": accuracy,
                "error_rate": error_rate,
                "num_errors": num_errors,
                "num_test": num_test,
            }
        )

        for index, (true_label, pred_label, distance, is_correct) in enumerate(
            zip(y_test, y_pred, distances, correct)
        ):
            prediction_rows.append(
                {
                    "k": int(k),
                    "test_path": test_paths[index] if test_paths else str(index),
                    "true_label": int(true_label),
                    "predicted_label": int(pred_label),
                    "distance": float(distance),
                    "correct": bool(is_correct),
                }
            )

    if not results:
        raise ValueError(f"Ninguna dimension solicitada es valida. Maximo disponible: {max_k}.")

    return results, prediction_rows


def save_mean_face(mean_face: np.ndarray, image_shape: tuple[int, int], output_path: str | Path) -> None:
    image = mean_face.reshape(image_shape)
    _write_grayscale(Path(output_path), to_uint8_visualization(image))


def save_eigenfaces_grid(
    eigenvectors: np.ndarray,
    image_shape: tuple[int, int],
    output_path: str | Path,
    num_faces: int = 16,
) -> None:
    if num_faces < 1:
        raise ValueError("num_faces debe ser positivo.")

    count = min(num_faces, eigenvectors.shape[1])
    cols = min(4, count)
    rows = int(np.ceil(count / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.4))
    axes_array = np.atleast_1d(axes).ravel()
    for index, ax in enumerate(axes_array):
        ax.axis("off")
        if index < count:
            face = eigenvectors[:, index].reshape(image_shape)
            ax.imshow(to_uint8_visualization(face), cmap="gray", vmin=0, vmax=255)
            ax.set_title(f"k={index + 1}")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_error_curve(results: list[dict[str, int | float]], output_path: str | Path) -> None:
    dims = [int(row["k"]) for row in results]
    errors = [float(row["error_rate"]) for row in results]

    plt.figure(figsize=(7, 5))
    plt.plot(dims, errors, marker="o", linewidth=1.8)
    plt.xlabel("d' (numero de eigenfaces)")
    plt.ylabel("Tasa de error")
    plt.title("Eigenfaces ORL: error frente a d'")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def write_csv(path: str | Path, rows: list[dict], fieldnames: list[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Eigenfaces para la base ORL.")
    parser.add_argument("--data-dir", required=True, help="Directorio raiz de ORL, por ejemplo orl/ o att_faces/.")
    parser.add_argument("--out-dir", default="results_eigenfaces", help="Directorio de salida.")
    parser.add_argument("--dims", default=DEFAULT_DIMS, help="Dimensiones d' separadas por comas.")
    parser.add_argument("--resize", default="0", help="0 para no redimensionar, o anchoxalto, por ejemplo 92x112.")
    parser.add_argument("--save-faces", action="store_true", help="Guarda una cuadricula con las primeras eigenfaces.")
    parser.add_argument("--num-faces", type=int, default=16, help="Numero de eigenfaces a dibujar si se usa --save-faces.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    dims = parse_dims(args.dims)
    resize = parse_resize(args.resize)
    if args.num_faces < 1:
        raise ValueError("--num-faces debe ser positivo.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    samples, labels = load_orl_dataset(args.data_dir, resize=resize)
    train_samples, y_train, test_samples, y_test = split_orl_train_test(samples, labels)
    image_shape = train_samples[0].image.shape

    X_train = vectorize_images(train_samples)
    X_test = vectorize_images(test_samples)
    mean_face, eigenvectors, eigvals = fit_eigenfaces(X_train)
    skipped_dims = [k for k in dims if k > eigenvectors.shape[1]]

    save_mean_face(mean_face, image_shape, out_dir / "mean_face.png")
    results, prediction_rows = evaluate_for_dimensions(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        mean_face=mean_face,
        eigenvectors=eigenvectors,
        dims=dims,
        test_paths=[str(sample.path) for sample in test_samples],
    )

    write_csv(
        out_dir / "results.csv",
        results,
        fieldnames=["k", "accuracy", "error_rate", "num_errors", "num_test"],
    )
    write_csv(
        out_dir / "predictions.csv",
        prediction_rows,
        fieldnames=["k", "test_path", "true_label", "predicted_label", "distance", "correct"],
    )
    plot_error_curve(results, out_dir / "error_curve.png")

    if args.save_faces:
        save_eigenfaces_grid(eigenvectors, image_shape, out_dir / "eigenfaces.png", num_faces=args.num_faces)

    print(f"Imagenes cargadas: {len(samples)}")
    print(f"Train: {len(train_samples)} | Test: {len(test_samples)} | Eigenfaces disponibles: {eigenvectors.shape[1]}")
    if skipped_dims:
        print(f"Dimensiones omitidas por superar el maximo disponible: {skipped_dims}")
    print(f"Salidas guardadas en: {out_dir}")
    print(f"Mejor accuracy: {max(row['accuracy'] for row in results):.4f}")
    print(f"Eigenvalor principal: {float(eigvals[0]):.6g}")


if __name__ == "__main__":
    main()
