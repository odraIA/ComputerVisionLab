import argparse
import csv
import os
import re
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
from scipy import linalg

try:
    import cv2
except ImportError:
    cv2 = None
    from PIL import Image


IMAGE_EXTENSIONS = {".pgm", ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
DEFAULT_LDA_DIMS = "1,2,3,5,10,15,20,25,30,35,39"
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


def parse_lda_dims(value: str) -> list[int]:
    dims = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not dims:
        raise ValueError("--lda-dims debe contener al menos una dimension.")
    if any(dim <= 0 for dim in dims):
        raise ValueError("Todas las dimensiones de --lda-dims deben ser enteros positivos.")
    return sorted(dict.fromkeys(dims))


def parse_pca_dim(value: str) -> int | str:
    if value.lower() == "auto":
        return "auto"
    pca_dim = int(value)
    if pca_dim <= 0:
        raise ValueError("--pca-dim debe ser 'auto' o un entero positivo.")
    return pca_dim


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


def _write_grayscale(path: str | Path, image: np.ndarray) -> None:
    path = Path(path)
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


def load_orl_dataset(
    root_dir: str | Path,
    resize: tuple[int, int] | None = None,
) -> tuple[list[ORLSample], np.ndarray]:
    root_dir = Path(root_dir)
    if not root_dir.exists():
        raise FileNotFoundError(f"No existe el directorio ORL: {root_dir}")

    samples: list[ORLSample] = []
    subject_dirs: list[tuple[int, Path]] = []
    for child in root_dir.iterdir():
        if child.is_dir():
            label = _subject_label(child.name)
            if label is not None:
                subject_dirs.append((label, child))

    for label, subject_dir in sorted(subject_dirs, key=lambda item: item[0]):
        image_paths: list[tuple[int, Path]] = []
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
    vectors: list[np.ndarray] = []
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


def fit_pca_small_matrix(
    X_train: np.ndarray,
    pca_dim: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X_train = np.asarray(X_train, dtype=np.float32)
    if X_train.ndim != 2:
        raise ValueError("X_train debe tener forma (n_imagenes, n_pixeles).")
    if pca_dim < 1:
        raise ValueError("pca_dim debe ser positivo.")

    n, _ = X_train.shape
    if n < 2:
        raise ValueError("Se necesitan al menos dos imagenes de entrenamiento.")

    mean_face = np.mean(X_train, axis=0, dtype=np.float64).astype(np.float32)
    A = X_train.astype(np.float64) - mean_face.astype(np.float64)

    C = (A @ A.T) / float(n - 1)
    eigvals, eigvecs_small = np.linalg.eigh(C)

    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs_small = eigvecs_small[:, order]

    tolerance = max(float(eigvals[0]) * 1e-9, EPS) if eigvals.size else EPS
    valid = eigvals > tolerance
    eigvals = eigvals[valid]
    eigvecs_small = eigvecs_small[:, valid]

    if eigvecs_small.size == 0:
        raise ValueError("No se pudieron obtener componentes PCA estables.")

    pca_dim = min(pca_dim, eigvecs_small.shape[1])
    eigvals = eigvals[:pca_dim]
    eigvecs_small = eigvecs_small[:, :pca_dim]

    W_pca = A.T @ eigvecs_small
    norms = np.linalg.norm(W_pca, axis=0)
    stable = norms > EPS
    W_pca = W_pca[:, stable] / norms[stable]
    eigvals = eigvals[stable]

    # Quito el cambio de signo aleatorio de los autovectores.
    for col in range(W_pca.shape[1]):
        pivot = int(np.argmax(np.abs(W_pca[:, col])))
        if W_pca[pivot, col] < 0:
            W_pca[:, col] *= -1.0

    if W_pca.shape[1] == 0:
        raise ValueError("Las componentes PCA calculadas no son numericamente estables.")

    return mean_face.astype(np.float32), W_pca.astype(np.float32), eigvals.astype(np.float32)


def fit_lda(
    Z_train_pca: np.ndarray,
    y_train: np.ndarray,
    lda_dim: int,
    reg: float,
) -> tuple[np.ndarray, np.ndarray]:
    Z_train_pca = np.asarray(Z_train_pca, dtype=np.float64)
    y_train = np.asarray(y_train)
    if Z_train_pca.ndim != 2:
        raise ValueError("Z_train_pca debe tener forma (n_imagenes, pca_dim).")
    if len(y_train) != Z_train_pca.shape[0]:
        raise ValueError("Z_train_pca e y_train deben tener el mismo numero de muestras.")
    if lda_dim < 1:
        raise ValueError("lda_dim debe ser positivo.")
    if reg < 0:
        raise ValueError("reg debe ser no negativo.")

    classes = np.unique(y_train)
    n_features = Z_train_pca.shape[1]
    max_dim = min(len(classes) - 1, n_features)
    if max_dim < 1:
        raise ValueError("LDA requiere al menos dos clases y una dimension PCA.")
    lda_dim = min(lda_dim, max_dim)

    mean_total = np.mean(Z_train_pca, axis=0)
    Sw = np.zeros((n_features, n_features), dtype=np.float64)
    Sb = np.zeros((n_features, n_features), dtype=np.float64)

    for label in classes:
        Z_class = Z_train_pca[y_train == label]
        mean_class = np.mean(Z_class, axis=0)
        centered = Z_class - mean_class
        Sw += centered.T @ centered
        mean_diff = (mean_class - mean_total).reshape(-1, 1)
        Sb += Z_class.shape[0] * (mean_diff @ mean_diff.T)

    Sw = 0.5 * (Sw + Sw.T)
    Sb = 0.5 * (Sb + Sb.T)

    identity = np.eye(n_features, dtype=np.float64)
    jitter = reg if reg > 0 else 1e-8
    last_error: Exception | None = None
    for attempt in range(7):
        Sw_reg = Sw + jitter * identity
        try:
            eigvals, eigvecs = linalg.eigh(Sb, Sw_reg, check_finite=False)
            break
        except linalg.LinAlgError as exc:
            last_error = exc
            jitter *= 10.0
    else:
        # Plan B por si Sw sale muy mal condicionada.
        if last_error is not None:
            Sw_reg = Sw + jitter * identity
        C = np.linalg.pinv(Sw_reg) @ Sb
        eigvals, eigvecs = np.linalg.eig(C)
        eigvals = np.real_if_close(eigvals, tol=1000)
        eigvecs = np.real_if_close(eigvecs, tol=1000)
        if np.iscomplexobj(eigvals) or np.iscomplexobj(eigvecs):
            imag_ok = np.max(np.abs(np.imag(eigvals))) < 1e-7 and np.max(np.abs(np.imag(eigvecs))) < 1e-7
            if not imag_ok:
                raise ValueError("El problema LDA produjo componentes complejas no despreciables.") from last_error
            eigvals = np.real(eigvals)
            eigvecs = np.real(eigvecs)

    eigvals = np.asarray(eigvals, dtype=np.float64)
    eigvecs = np.asarray(eigvecs, dtype=np.float64)
    finite = np.isfinite(eigvals) & np.all(np.isfinite(eigvecs), axis=0)
    eigvals = eigvals[finite]
    eigvecs = eigvecs[:, finite]
    if eigvecs.size == 0:
        raise ValueError("No se pudieron obtener vectores LDA finitos.")

    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    eigvals = eigvals[:lda_dim]
    V_lda = eigvecs[:, :lda_dim]

    norms = np.linalg.norm(V_lda, axis=0)
    stable = norms > EPS
    V_lda = V_lda[:, stable] / norms[stable]
    eigvals = eigvals[stable]

    for col in range(V_lda.shape[1]):
        pivot = int(np.argmax(np.abs(V_lda[:, col])))
        if V_lda[pivot, col] < 0:
            V_lda[:, col] *= -1.0

    if V_lda.shape[1] == 0:
        raise ValueError("Las componentes LDA calculadas no son numericamente estables.")

    return V_lda.astype(np.float32), eigvals.astype(np.float32)


def project_fisherfaces(
    X: np.ndarray,
    mean_face: np.ndarray,
    W_pca: np.ndarray,
    V_lda: np.ndarray,
) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    return ((X - mean_face) @ W_pca) @ V_lda


def nearest_neighbor_predict(
    Z_train: np.ndarray,
    y_train: np.ndarray,
    Z_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    Z_train = np.asarray(Z_train, dtype=np.float32)
    Z_test = np.asarray(Z_test, dtype=np.float32)
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
    W_pca: np.ndarray,
    V_lda: np.ndarray,
    lda_dims: list[int],
    test_paths: list[str] | None = None,
) -> tuple[list[dict[str, int | float]], list[dict[str, int | float | str | bool]]]:
    max_dim = V_lda.shape[1]
    valid_dims = sorted({min(dim, max_dim) for dim in lda_dims if dim > 0})
    if not valid_dims:
        raise ValueError(f"Ninguna dimension LDA solicitada es valida. Maximo disponible: {max_dim}.")

    results: list[dict[str, int | float]] = []
    prediction_rows: list[dict[str, int | float | str | bool]] = []
    pca_dim = W_pca.shape[1]

    for lda_dim in valid_dims:
        V_partial = V_lda[:, :lda_dim]
        Z_train = project_fisherfaces(X_train, mean_face, W_pca, V_partial)
        Z_test = project_fisherfaces(X_test, mean_face, W_pca, V_partial)
        y_pred, distances = nearest_neighbor_predict(Z_train, y_train, Z_test)
        correct = y_pred == y_test
        num_errors = int(np.sum(~correct))
        num_test = int(len(y_test))
        accuracy = float(np.mean(correct))
        error_rate = float(num_errors / num_test)

        results.append(
            {
                "lda_dim": int(lda_dim),
                "pca_dim": int(pca_dim),
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
                    "lda_dim": int(lda_dim),
                    "pca_dim": int(pca_dim),
                    "test_path": test_paths[index] if test_paths else str(index),
                    "true_label": int(true_label),
                    "predicted_label": int(pred_label),
                    "distance": float(distance),
                    "correct": bool(is_correct),
                }
            )

    return results, prediction_rows


def plot_error_curve(results: list[dict[str, int | float]], output_path: str | Path) -> None:
    dims = [int(row["lda_dim"]) for row in results]
    errors = [float(row["error_rate"]) for row in results]

    plt.figure(figsize=(7, 5))
    plt.plot(dims, errors, marker="o", linewidth=1.8)
    plt.xlabel("d' (dimensiones LDA)")
    plt.ylabel("Tasa de error")
    plt.title("Fisherfaces ORL: error frente a d'")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def save_mean_face(mean_face: np.ndarray, image_shape: tuple[int, int], output_path: str | Path) -> None:
    image = mean_face.reshape(image_shape)
    _write_grayscale(output_path, to_uint8_visualization(image))


def save_fisherfaces_visualization(
    W_pca: np.ndarray,
    V_lda: np.ndarray,
    image_shape: tuple[int, int],
    output_path: str | Path,
    num_faces: int = 16,
) -> None:
    if num_faces < 1:
        raise ValueError("num_faces debe ser positivo.")

    fisherfaces = W_pca @ V_lda
    count = min(num_faces, fisherfaces.shape[1])
    cols = min(4, count)
    rows = int(np.ceil(count / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.4))
    axes_array = np.atleast_1d(axes).ravel()
    for index, ax in enumerate(axes_array):
        ax.axis("off")
        if index < count:
            face = fisherfaces[:, index].reshape(image_shape)
            ax.imshow(to_uint8_visualization(face), cmap="gray", vmin=0, vmax=255)
            ax.set_title(f"d'={index + 1}")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def write_csv(path: str | Path, rows: list[dict], fieldnames: list[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def resolve_pca_dim(pca_dim_arg: int | str, n_train: int, n_pixels: int, num_classes: int) -> int:
    if pca_dim_arg == "auto":
        return min(max(n_train - num_classes, 1), n_pixels, 150)
    return min(int(pca_dim_arg), n_train - 1, n_pixels)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fisherfaces para la base ORL.")
    parser.add_argument("--data-dir", required=True, help="Directorio raiz de ORL, por ejemplo orl/ o att_faces/.")
    parser.add_argument("--out-dir", default="results_fisherfaces", help="Directorio de salida.")
    parser.add_argument("--lda-dims", default=DEFAULT_LDA_DIMS, help="Dimensiones LDA d' separadas por comas.")
    parser.add_argument("--pca-dim", default="auto", help="'auto' o dimension PCA intermedia positiva.")
    parser.add_argument("--reg", type=float, default=1e-6, help="Regularizacion diagonal para Sw.")
    parser.add_argument("--resize", default="0", help="0 para no redimensionar, o anchoxalto, por ejemplo 92x112.")
    parser.add_argument("--save-faces", action="store_true", help="Guarda una cuadricula de fisherfaces aproximadas.")
    parser.add_argument("--num-faces", type=int, default=16, help="Numero de fisherfaces a dibujar si se usa --save-faces.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    lda_dims = parse_lda_dims(args.lda_dims)
    pca_dim_arg = parse_pca_dim(args.pca_dim)
    resize = parse_resize(args.resize)
    if args.reg < 0:
        raise ValueError("--reg debe ser no negativo.")
    if args.num_faces < 1:
        raise ValueError("--num-faces debe ser positivo.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    samples, labels = load_orl_dataset(args.data_dir, resize=resize)
    train_samples, y_train, test_samples, y_test = split_orl_train_test(samples, labels)
    image_shape = train_samples[0].image.shape

    X_train = vectorize_images(train_samples)
    X_test = vectorize_images(test_samples)
    classes = np.unique(y_train)
    pca_dim_requested = resolve_pca_dim(
        pca_dim_arg=pca_dim_arg,
        n_train=X_train.shape[0],
        n_pixels=X_train.shape[1],
        num_classes=len(classes),
    )
    mean_face, W_pca, pca_eigvals = fit_pca_small_matrix(X_train, pca_dim_requested)
    Z_train_pca = (X_train - mean_face) @ W_pca

    max_lda_dim = min(max(lda_dims), len(classes) - 1, W_pca.shape[1])
    V_lda, lda_eigvals = fit_lda(Z_train_pca, y_train, lda_dim=max_lda_dim, reg=args.reg)
    effective_max_dim = V_lda.shape[1]
    skipped_dims = [dim for dim in lda_dims if dim > effective_max_dim]

    save_mean_face(mean_face, image_shape, out_dir / "mean_face.png")
    results, prediction_rows = evaluate_for_dimensions(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        mean_face=mean_face,
        W_pca=W_pca,
        V_lda=V_lda,
        lda_dims=lda_dims,
        test_paths=[str(sample.path) for sample in test_samples],
    )

    write_csv(
        out_dir / "results.csv",
        results,
        fieldnames=["lda_dim", "pca_dim", "accuracy", "error_rate", "num_errors", "num_test"],
    )
    write_csv(
        out_dir / "predictions.csv",
        prediction_rows,
        fieldnames=[
            "lda_dim",
            "pca_dim",
            "test_path",
            "true_label",
            "predicted_label",
            "distance",
            "correct",
        ],
    )
    plot_error_curve(results, out_dir / "error_curve.png")

    if args.save_faces:
        save_fisherfaces_visualization(
            W_pca=W_pca,
            V_lda=V_lda,
            image_shape=image_shape,
            output_path=out_dir / "fisherfaces.png",
            num_faces=args.num_faces,
        )

    print(f"Imagenes cargadas: {len(samples)}")
    print(f"Train: {len(train_samples)} | Test: {len(test_samples)} | Clases: {len(classes)}")
    print(f"PCA solicitada: {pca_dim_requested} | PCA efectiva: {W_pca.shape[1]}")
    print(f"LDA maxima efectiva: {effective_max_dim} | Regularizacion Sw: {args.reg:g}")
    if skipped_dims:
        print(f"Dimensiones LDA ajustadas al maximo efectivo ({effective_max_dim}): {skipped_dims}")
    print(f"Salidas guardadas en: {out_dir}")
    print(f"Mejor accuracy: {max(row['accuracy'] for row in results):.4f}")
    print(f"Eigenvalor PCA principal: {float(pca_eigvals[0]):.6g}")
    print(f"Eigenvalor LDA principal: {float(lda_eigvals[0]):.6g}")


if __name__ == "__main__":
    main()
