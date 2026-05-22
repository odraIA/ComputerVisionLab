import argparse
import csv
import os
import re
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None
    from PIL import Image


BASE_DIR = Path(__file__).resolve().parent
YALEB_DIR = BASE_DIR / "ILLUMINATION" / "YaleB"
EXTENSIONES = {".pgm", ".png", ".jpg", ".jpeg", ".bmp"}
METODOS = ("norm_global", "hist_global", "norm_local", "hist_local")
ALIAS_METODOS = {
    "global_norm": "norm_global",
    "global_histeq": "hist_global",
    "local_norm": "norm_local",
    "local_histeq": "hist_local",
}
TITULOS = {
    "norm_global": "Norm. global",
    "hist_global": "Hist. global",
    "norm_local": "Norm. local",
    "hist_local": "Hist. local",
}


def leer_gris(ruta: str | Path) -> np.ndarray:
    if cv2 is not None:
        img = cv2.imread(str(ruta), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"No se pudo leer la imagen: {ruta}")
    else:
        with Image.open(ruta) as entrada:
            img = np.asarray(entrada.convert("L"))
    return img.astype(np.float32)


def guardar_gris(ruta: str | Path, img: np.ndarray) -> None:
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    img_u8 = a_uint8(img)
    if cv2 is not None:
        cv2.imwrite(str(ruta), img_u8)
    else:
        Image.fromarray(img_u8).save(ruta)


def a_uint8(img: np.ndarray) -> np.ndarray:
    img = np.asarray(img, dtype=np.float32)
    minimo = float(img.min())
    maximo = float(img.max())
    if maximo - minimo < 1e-12:
        return np.zeros(img.shape, dtype=np.uint8)

    img = (img - minimo) * (255.0 / (maximo - minimo))
    return np.clip(img, 0, 255).astype(np.uint8)


def norm_global(img: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    img = img.astype(np.float32, copy=False)
    return (img - img.mean()) / (img.std() + eps)


def hist_global(img: np.ndarray) -> np.ndarray:
    img_u8 = np.clip(img, 0, 255).astype(np.uint8)
    hist = np.bincount(img_u8.ravel(), minlength=256).astype(np.float32)
    cdf = hist.cumsum()
    tabla = np.floor(255.0 * cdf / img_u8.size)
    return tabla[img_u8].astype(np.float32)


def integral(img: np.ndarray) -> np.ndarray:
    img = img.astype(np.float64, copy=False)
    out = np.zeros((img.shape[0] + 1, img.shape[1] + 1), dtype=np.float64)
    out[1:, 1:] = img.cumsum(axis=0).cumsum(axis=1)
    return out


def suma_rect(ii: np.ndarray, y0: int, x0: int, y1: int, x1: int) -> float:
    return float(ii[y1, x1] - ii[y0, x1] - ii[y1, x0] + ii[y0, x0])


def media_var_local(img: np.ndarray, lado: int) -> tuple[np.ndarray, np.ndarray]:
    if lado < 1:
        raise ValueError("lado debe ser >= 1")

    img = img.astype(np.float32, copy=False)
    h, w = img.shape
    r = lado // 2

    yy = np.arange(h)
    xx = np.arange(w)
    y0 = np.maximum(0, yy - r)
    y1 = np.minimum(h, yy + r + 1)
    x0 = np.maximum(0, xx - r)
    x1 = np.minimum(w, xx + r + 1)

    ii = integral(img)
    ii2 = integral(img * img)

    suma = ii[y1[:, None], x1[None, :]] - ii[y0[:, None], x1[None, :]]
    suma -= ii[y1[:, None], x0[None, :]]
    suma += ii[y0[:, None], x0[None, :]]

    suma2 = ii2[y1[:, None], x1[None, :]] - ii2[y0[:, None], x1[None, :]]
    suma2 -= ii2[y1[:, None], x0[None, :]]
    suma2 += ii2[y0[:, None], x0[None, :]]

    area = (y1 - y0)[:, None] * (x1 - x0)[None, :]
    media = suma / area
    var = np.maximum(suma2 / area - media * media, 0.0)
    return media.astype(np.float32), var.astype(np.float32)


def inicios(longitud: int, lado: int, paso: int) -> list[int]:
    if paso < 1:
        raise ValueError("paso debe ser >= 1")
    if lado >= longitud:
        return [0]

    puntos = list(range(0, longitud - lado + 1, paso))
    ultimo = longitud - lado
    if puntos[-1] != ultimo:
        puntos.append(ultimo)
    return puntos


def norm_local(img: np.ndarray, lado: int, eps: float = 1e-6, paso: int = 1) -> np.ndarray:
    if lado < 1:
        raise ValueError("lado debe ser >= 1")

    img = img.astype(np.float32, copy=False)
    h, w = img.shape
    lado_y = min(lado, h)
    lado_x = min(lado, w)
    ii = integral(img)
    ii2 = integral(img * img)
    acc = np.zeros_like(img, dtype=np.float32)
    n = np.zeros_like(img, dtype=np.float32)

    for y0 in inicios(h, lado_y, paso):
        y1 = y0 + lado_y
        for x0 in inicios(w, lado_x, paso):
            x1 = x0 + lado_x
            area = float((y1 - y0) * (x1 - x0))
            total = suma_rect(ii, y0, x0, y1, x1)
            total2 = suma_rect(ii2, y0, x0, y1, x1)
            media = total / area
            var = max(total2 / area - media * media, 0.0)
            acc[y0:y1, x0:x1] += (img[y0:y1, x0:x1] - media) / (np.sqrt(var) + eps)
            n[y0:y1, x0:x1] += 1.0

    return acc / np.maximum(n, 1.0)


def ecualizar_parche(parche: np.ndarray, clip: float | None) -> np.ndarray:
    parche_u8 = np.clip(parche, 0, 255).astype(np.uint8)
    hist = np.bincount(parche_u8.ravel(), minlength=256).astype(np.float32)

    if clip is not None and clip > 0:
        techo = max(1.0, clip * parche_u8.size / 256.0)
        exceso = np.maximum(hist - techo, 0.0)
        hist = np.minimum(hist, techo)
        hist += exceso.sum() / 256.0

    cdf = hist.cumsum()
    tabla = np.floor(255.0 * cdf / parche_u8.size)
    return tabla[parche_u8].astype(np.float32)


def hist_local(img: np.ndarray, lado: int, clip: float | None = None, paso: int = 1) -> np.ndarray:
    if lado < 1:
        raise ValueError("lado debe ser >= 1")

    img = img.astype(np.float32, copy=False)
    h, w = img.shape
    lado_y = min(lado, h)
    lado_x = min(lado, w)
    acc = np.zeros_like(img, dtype=np.float32)
    n = np.zeros_like(img, dtype=np.float32)

    for y0 in inicios(h, lado_y, paso):
        y1 = y0 + lado_y
        for x0 in inicios(w, lado_x, paso):
            x1 = x0 + lado_x
            acc[y0:y1, x0:x1] += ecualizar_parche(img[y0:y1, x0:x1], clip)
            n[y0:y1, x0:x1] += 1.0

    return acc / np.maximum(n, 1.0)


def metodo_normalizado(nombre: str) -> str:
    nombre = ALIAS_METODOS.get(nombre, nombre)
    if nombre not in METODOS:
        raise ValueError(f"Metodo desconocido: {nombre}")
    return nombre


def lista_metodos(nombre: str) -> list[str]:
    return list(METODOS) if nombre == "all" else [metodo_normalizado(nombre)]


def aplicar_metodo(
    img: np.ndarray,
    metodo: str,
    lado: int,
    paso: int,
    clip: float,
    eps: float,
) -> np.ndarray:
    metodo = metodo_normalizado(metodo)
    if metodo == "norm_global":
        return norm_global(img, eps=eps)
    if metodo == "hist_global":
        return hist_global(img)
    if metodo == "norm_local":
        return norm_local(img, lado=lado, eps=eps, paso=paso)
    return hist_local(img, lado=lado, clip=clip if clip > 0 else None, paso=paso)


def clave_yaleb(ruta: Path) -> str:
    match = re.match(r"(yaleB\d+_P\d+A[+-]\d+E[+-]\d+)", ruta.stem)
    return match.group(1) if match else ruta.stem


def imagenes_yaleb(raiz: str | Path = YALEB_DIR) -> list[Path]:
    raiz = Path(raiz)
    if not raiz.exists():
        raise FileNotFoundError(f"No existe YaleB en: {raiz}")

    elegidas: dict[str, Path] = {}
    for ruta in sorted(raiz.rglob("*")):
        if not ruta.is_file() or ruta.suffix.lower() not in EXTENSIONES:
            continue

        clave = clave_yaleb(ruta)
        anterior = elegidas.get(clave)
        if anterior is None or (anterior.suffix.lower() != ".pgm" and ruta.suffix.lower() == ".pgm"):
            elegidas[clave] = ruta

    if not elegidas:
        raise ValueError(f"No se encontraron imagenes YaleB en {raiz}")
    return sorted(elegidas.values())


def imagenes_entrada(ruta: str | Path) -> list[Path]:
    ruta = Path(ruta)
    if ruta.is_file():
        return [ruta]
    if ruta.is_dir():
        return imagenes_yaleb(ruta)
    raise FileNotFoundError(f"No existe la entrada: {ruta}")


def nombre_salida(ruta: Path, raiz: Path | None, metodo: str) -> Path:
    if raiz is not None:
        try:
            rel = ruta.relative_to(raiz)
            base = Path(*rel.parts[:-1]) / rel.stem
        except ValueError:
            base = Path(ruta.stem)
    else:
        base = Path(ruta.stem)

    return base.with_name(f"{base.name}_{metodo}.png")


def procesar_imagen(
    ruta_imagen: str | Path,
    carpeta_salida: str | Path,
    metodo: str = "all",
    lado: int = 15,
    paso: int = 1,
    clip: float = 0.0,
    eps: float = 1e-6,
    guardar_comparativa: bool = False,
    raiz_entrada: Path | None = None,
) -> list[dict[str, str | int | float]]:
    ruta_imagen = Path(ruta_imagen)
    carpeta_salida = Path(carpeta_salida)
    img = leer_gris(ruta_imagen)
    filas: list[dict[str, str | int | float]] = []
    resultados: dict[str, np.ndarray] = {}

    for metodo_actual in lista_metodos(metodo):
        t0 = time.perf_counter()
        procesada = aplicar_metodo(img, metodo_actual, lado, paso, clip, eps)
        segundos = time.perf_counter() - t0

        salida = carpeta_salida / nombre_salida(ruta_imagen, raiz_entrada, metodo_actual)
        guardar_gris(salida, procesada)

        resultados[metodo_actual] = procesada
        filas.append(
            {
                "imagen": str(ruta_imagen),
                "metodo": metodo_actual,
                "lado": lado,
                "paso": paso,
                "segundos": segundos,
                "salida": str(salida),
            }
        )

    if guardar_comparativa:
        guardar_panel(img, resultados, ruta_imagen, carpeta_salida, lado, paso, clip, eps, raiz_entrada)

    return filas


def guardar_panel(
    original: np.ndarray,
    resultados: dict[str, np.ndarray],
    ruta_imagen: Path,
    carpeta_salida: Path,
    lado: int,
    paso: int,
    clip: float,
    eps: float,
    raiz_entrada: Path | None,
) -> None:
    import matplotlib.pyplot as plt

    for metodo in METODOS:
        if metodo not in resultados:
            resultados[metodo] = aplicar_metodo(original, metodo, lado, paso, clip, eps)

    fig, axes = plt.subplots(1, 5, figsize=(16, 4))
    nombres = ["Original"] + [TITULOS[metodo] for metodo in METODOS]
    imagenes = [original] + [resultados[metodo] for metodo in METODOS]

    for ax, titulo, img in zip(axes, nombres, imagenes):
        ax.imshow(a_uint8(img), cmap="gray", vmin=0, vmax=255)
        ax.set_title(titulo)
        ax.axis("off")

    fig.tight_layout()
    salida = carpeta_salida / nombre_salida(ruta_imagen, raiz_entrada, "comparativa")
    salida.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(salida, dpi=150)
    plt.close(fig)


def procesar_carpeta(
    entrada: str | Path = YALEB_DIR,
    carpeta_salida: str | Path = BASE_DIR / "results_light_yaleb",
    metodo: str = "all",
    lado: int = 15,
    paso: int = 1,
    clip: float = 0.0,
    eps: float = 1e-6,
    guardar_comparativa: bool = False,
    limite: int | None = None,
) -> list[dict[str, str | int | float]]:
    entrada = Path(entrada)
    imagenes = imagenes_entrada(entrada)
    if limite is not None:
        imagenes = imagenes[:limite]

    raiz = entrada if entrada.is_dir() else None
    filas: list[dict[str, str | int | float]] = []
    for ruta in imagenes:
        filas.extend(
            procesar_imagen(
                ruta_imagen=ruta,
                carpeta_salida=carpeta_salida,
                metodo=metodo,
                lado=lado,
                paso=paso,
                clip=clip,
                eps=eps,
                guardar_comparativa=guardar_comparativa,
                raiz_entrada=raiz,
            )
        )

    resumen = Path(carpeta_salida) / "summary.csv"
    resumen.parent.mkdir(parents=True, exist_ok=True)
    with resumen.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["imagen", "metodo", "lado", "paso", "segundos", "salida"])
        writer.writeheader()
        writer.writerows(filas)

    return filas


def parse_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Pruebas de normalizacion de luz en YaleB.")
    parser.add_argument("--input", default=str(YALEB_DIR), help="Imagen o carpeta. Por defecto usa YaleB.")
    parser.add_argument("--out-dir", default=str(BASE_DIR / "results_light_yaleb"), help="Carpeta de salida.")
    parser.add_argument(
        "--method",
        choices=(*METODOS, *ALIAS_METODOS.keys(), "all"),
        default="all",
        help="Metodo a aplicar.",
    )
    parser.add_argument("--window-size", type=int, default=15, help="Tamano de la ventana local.")
    parser.add_argument("--stride", type=int, default=1, help="Paso entre ventanas solapadas.")
    parser.add_argument("--clip-limit", type=float, default=0.0, help="Recorte relativo para hist_local; 0 lo desactiva.")
    parser.add_argument("--save-comparison", action="store_true", help="Guarda una comparativa por imagen.")
    parser.add_argument("--limit", type=int, default=None, help="Procesa solo las primeras N imagenes.")
    return parser


def main() -> None:
    args = parse_args().parse_args()
    filas = procesar_carpeta(
        entrada=args.input,
        carpeta_salida=args.out_dir,
        metodo=args.method,
        lado=args.window_size,
        paso=args.stride,
        clip=args.clip_limit,
        guardar_comparativa=args.save_comparison,
        limite=args.limit,
    )
    resumen = Path(args.out_dir) / "summary.csv"
    print(f"Procesadas {len(filas)} combinaciones imagen/metodo. Resumen: {resumen}")


# Nombres que use al principio en el notebook.
IMAGE_EXTENSIONS = EXTENSIONES
METHODS = ("global_norm", "global_histeq", "local_norm", "local_histeq")
METODO_ANTIGUO = {
    "norm_global": "global_norm",
    "hist_global": "global_histeq",
    "norm_local": "local_norm",
    "hist_local": "local_histeq",
}
read_grayscale = leer_gris
to_uint8_visualization = a_uint8
global_normalization = norm_global
global_hist_equalization = hist_global
integral_image = integral
build_parser = parse_args


def local_mean_variance_integral(image: np.ndarray, window_size: int) -> tuple[np.ndarray, np.ndarray]:
    return media_var_local(image, lado=window_size)


def local_normalization(
    image: np.ndarray,
    window_size: int,
    eps: float = 1e-6,
    stride: int = 1,
) -> np.ndarray:
    return norm_local(image, lado=window_size, eps=eps, paso=stride)


def local_hist_equalization(
    image: np.ndarray,
    window_size: int,
    bins: int = 256,
    clip_limit: float | None = None,
    stride: int = 1,
) -> np.ndarray:
    if bins != 256:
        raise ValueError("Esta version trabaja con imagenes de 8 bits; bins debe ser 256.")
    return hist_local(image, lado=window_size, clip=clip_limit, paso=stride)


def process_image(
    image_path: str | Path,
    out_dir: str | Path,
    method: str = "all",
    window_size: int = 15,
    stride: int = 1,
    clip_limit: float = 0.0,
    eps: float = 1e-6,
    save_comparison: bool = False,
) -> list[dict[str, str | int | float]]:
    filas = procesar_imagen(
        ruta_imagen=image_path,
        carpeta_salida=out_dir,
        metodo=method,
        lado=window_size,
        paso=stride,
        clip=clip_limit,
        eps=eps,
        guardar_comparativa=save_comparison,
    )
    return [_fila_formato_antiguo(fila) for fila in filas]


def process_directory(
    input_path: str | Path = YALEB_DIR,
    out_dir: str | Path = BASE_DIR / "results_light_yaleb",
    method: str = "all",
    window_size: int = 15,
    stride: int = 1,
    clip_limit: float = 0.0,
    eps: float = 1e-6,
    save_comparison: bool = False,
) -> list[dict[str, str | int | float]]:
    filas = procesar_carpeta(
        entrada=input_path,
        carpeta_salida=out_dir,
        metodo=method,
        lado=window_size,
        paso=stride,
        clip=clip_limit,
        eps=eps,
        guardar_comparativa=save_comparison,
    )
    filas_antiguas = [_fila_formato_antiguo(fila) for fila in filas]

    resumen = Path(out_dir) / "summary.csv"
    with resumen.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["image", "method", "window_size", "stride", "time_seconds", "output_path"],
        )
        writer.writeheader()
        writer.writerows(filas_antiguas)

    return filas_antiguas


def _fila_formato_antiguo(fila: dict[str, str | int | float]) -> dict[str, str | int | float]:
    return {
        "image": Path(str(fila["imagen"])).name,
        "method": METODO_ANTIGUO.get(str(fila["metodo"]), str(fila["metodo"])),
        "window_size": fila["lado"],
        "stride": fila["paso"],
        "time_seconds": fila["segundos"],
        "output_path": fila["salida"],
    }


if __name__ == "__main__":
    main()
