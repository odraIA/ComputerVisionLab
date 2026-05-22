import argparse
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd


DEMO_CLIENTS = np.array([0.9, 0.7, 0.7, 0.8, 0.4], dtype=float)
DEMO_IMPOSTORS = np.array([0.2, 0.1, 0.5, 0.5, 0.3], dtype=float)


def read_scores(path: str | Path) -> np.ndarray:
    text = Path(path).read_text(encoding="utf-8")
    rows = [
        [token for token in re.split(r"[\s,;]+", line.strip()) if token]
        for line in text.splitlines()
        if line.strip()
    ]

    if rows and all(len(row) == 2 for row in rows):
        tokens = [row[1] for row in rows]
    else:
        tokens = [token for token in re.split(r"[\s,;]+", text.strip()) if token]

    if not tokens:
        raise ValueError(f"No hay scores en {path}")

    try:
        scores = np.array([float(token) for token in tokens], dtype=float)
    except ValueError as exc:
        raise ValueError(f"Hay algun score no numerico en {path}") from exc

    if not np.all(np.isfinite(scores)):
        raise ValueError(f"Todos los scores tienen que ser finitos en {path}")
    return scores


def build_thresholds(clients: np.ndarray, impostors: np.ndarray) -> np.ndarray:
    all_scores = np.concatenate([clients, impostors])
    min_score = float(np.min(all_scores))
    max_score = float(np.max(all_scores))

    thresholds = set(float(score) for score in all_scores)
    thresholds.add(float(np.nextafter(min_score, -np.inf)))
    thresholds.add(float(np.nextafter(max_score, np.inf)))

    if min_score >= 0.0 and max_score <= 1.0:
        thresholds.add(0.0)
        thresholds.add(1.0)

    return np.array(sorted(thresholds), dtype=float)


def compute_roc_points(clients: np.ndarray, impostors: np.ndarray) -> pd.DataFrame:
    thresholds = build_thresholds(clients, impostors)
    total_clients = len(clients)
    total_impostors = len(impostors)

    rows = []
    for threshold in thresholds:
        vp = int(np.sum(clients >= threshold))
        fn = total_clients - vp
        fp = int(np.sum(impostors >= threshold))
        vn = total_impostors - fp

        tpr = vp / total_clients
        tnr = vn / total_impostors
        fpr = fp / total_impostors
        fnr = fn / total_clients

        rows.append(
            {
                "threshold": threshold,
                "VP": vp,
                "VN": vn,
                "FP": fp,
                "FN": fn,
                "TPR": tpr,
                "S": tpr,
                "FPR": fpr,
                "FNR": fnr,
                "TNR": tnr,
                "E": tnr,
            }
        )

    return pd.DataFrame(rows)


def auc_clientes_impostores(clients: np.ndarray, impostors: np.ndarray) -> float:
    if len(clients) == 0 or len(impostors) == 0:
        raise ValueError("Debe haber al menos un score de cliente y uno de impostor.")

    total = 0.0

    for client_score in clients:
        for impostor_score in impostors:
            if client_score > impostor_score:
                total += 1.0
            elif client_score == impostor_score:
                total += 0.5

    return total / (len(clients) * len(impostors))


def auc(clients: np.ndarray, impostors: np.ndarray) -> float:
    return auc_clientes_impostores(clients, impostors)


def auc_trapezoidal(clients: np.ndarray, impostors: np.ndarray | None = None) -> float:
    if impostors is None:
        raise TypeError("Ahora el AUC se calcula con clientes e impostores, no con puntos ROC.")
    return auc_clientes_impostores(clients, impostors)


def d_prime(clients: np.ndarray, impostors: np.ndarray) -> float:
    mu_clients = float(np.mean(clients))
    mu_impostors = float(np.mean(impostors))
    var_clients = float(np.var(clients))
    var_impostors = float(np.var(impostors))
    denominator = math.sqrt(var_clients + var_impostors)
    if denominator == 0.0:
        if mu_clients == mu_impostors:
            return 0.0
        return float("inf") if mu_clients > mu_impostors else float("-inf")
    return (mu_clients - mu_impostors) / denominator


def row_to_dict(row: pd.Series) -> dict[str, float | int]:
    enteros = {"VP", "VN", "FP", "FN"}
    salida = {}
    for key, value in row.items():
        if key in enteros:
            salida[key] = int(value)
        elif isinstance(value, (np.integer, int)):
            salida[key] = int(value)
        else:
            salida[key] = float(value)
    return salida


def closest_point(points: pd.DataFrame, metric: str, target: float) -> dict:
    i = (points[metric] - target).abs().idxmin()
    return row_to_dict(points.loc[i])


def best_with_limit(
    points: pd.DataFrame,
    limit_metric: str,
    limit_value: float,
    optimize_metric: str,
) -> dict | None:
    mejor_fila = None
    mejor_clave = None

    for _, row in points.iterrows():
        if row[limit_metric] > limit_value:
            continue

        clave = (
            float(row[optimize_metric]),
            abs(float(row[limit_metric]) - limit_value),
            float(row["threshold"]),
        )
        if mejor_clave is None or clave < mejor_clave:
            mejor_clave = clave
            mejor_fila = row

    if mejor_fila is None:
        return None
    return row_to_dict(mejor_fila)


def approximate_eer(points: pd.DataFrame) -> dict:
    mejor = None
    mejor_dif = None

    for _, row in points.iterrows():
        dif = abs(float(row["FPR"]) - float(row["FNR"]))
        if mejor_dif is None:
            mejor_dif = dif
            mejor = row
            continue

        if dif < mejor_dif:
            mejor_dif = dif
            mejor = row
        elif dif == mejor_dif and row["threshold"] < mejor["threshold"]:
            mejor = row

    salida = row_to_dict(mejor)
    salida["abs_difference"] = float(mejor_dif)
    salida["eer"] = float((mejor["FPR"] + mejor["FNR"]) / 2.0)
    return salida


def plot_roc(points: pd.DataFrame, output_path: Path | None = None) -> None:
    import os

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib.pyplot as plt

    roc = points.sort_values(["FPR", "TPR"], ascending=[True, True])
    roc = roc.drop_duplicates(subset=["FPR", "TPR"])

    plt.figure(figsize=(7, 6))
    line = plt.step(roc["FPR"], roc["TPR"], where="post", linewidth=1.8, label="ROC")[0]
    plt.plot(roc["FPR"], roc["TPR"], "o", markersize=4, color=line.get_color())
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Azar")
    plt.xlabel("FPR / FP")
    plt.ylabel("TPR / Sensibilidad")
    plt.title("Curva ROC")
    plt.xlim(-0.02, 1.02)
    plt.ylim(-0.02, 1.02)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    if output_path is not None:
        plt.savefig(output_path, dpi=150)
    plt.close()


def format_point(point: dict | None) -> str:
    if point is None:
        return "No existe punto que cumpla la restriccion."
    return (
        f"threshold={point['threshold']:.10g}, "
        f"FPR={point['FPR']:.6f}, FNR={point['FNR']:.6f}, "
        f"TPR={point['TPR']:.6f}, TNR={point['TNR']:.6f}, "
        f"VP={point['VP']}, VN={point['VN']}, FP={point['FP']}, FN={point['FN']}"
    )


def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        if value > 0:
            return "Infinity"
        if value < 0:
            return "-Infinity"
        return "NaN"
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calculos basicos de ROC para los scores de la practica."
    )
    parser.add_argument("--clients", help="Fichero con scores de clientes.")
    parser.add_argument("--impostors", help="Fichero con scores de impostores.")
    parser.add_argument("--fn-x", type=float, default=0.1, help="Valor objetivo de FNR.")
    parser.add_argument("--fp-x", type=float, default=0.05, help="Valor objetivo de FPR.")
    parser.add_argument("--out-dir", default="results_roc", help="Directorio de salida.")
    parser.add_argument("--demo", action="store_true", help="Usa el ejemplo integrado.")
    return parser.parse_args()


def load_input_scores(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    if args.demo:
        return DEMO_CLIENTS.copy(), DEMO_IMPOSTORS.copy()

    if not args.clients or not args.impostors:
        raise ValueError("Debes indicar --clients y --impostors, o usar --demo.")

    return read_scores(args.clients), read_scores(args.impostors)


def validate_inputs(clients: np.ndarray, impostors: np.ndarray, fn_x: float, fp_x: float) -> None:
    if len(clients) == 0 or len(impostors) == 0:
        raise ValueError("Debe haber al menos un score de cliente y uno de impostor.")
    if not 0.0 <= fn_x <= 1.0:
        raise ValueError("--fn-x debe estar entre 0 y 1.")
    if not 0.0 <= fp_x <= 1.0:
        raise ValueError("--fp-x debe estar entre 0 y 1.")
