"""Build the SOP v2 12-feature tables from discharge-capacity (QD) series only."""

from __future__ import annotations

import argparse
import math
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import skew


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
INTERMEDIATE_DIR = PROJECT_ROOT / "data" / "intermediate"
DEFAULT_HUST_DIR = Path("/Users/diclesaracoban/Downloads/HUST_data")
DEFAULT_WINDOWS = (50, 100)

SOP12_FEATURES = [
    "Qdis_N",
    "delta_Qdis",
    "retention_ratio",
    "slope_linear",
    "variance_Qdis",
    "range_Qdis",
    "max_drop",
    "std_diff",
    "skewness_Qdis",
    "slope_ratio",
    "Qdis_cycle10",
    "mean_diff",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build SOP v2 12-feature tables.")
    parser.add_argument("--hust-dir", type=Path, default=DEFAULT_HUST_DIR)
    parser.add_argument("--labels", type=Path, default=INTERMEDIATE_DIR / "raw_label_table.csv")
    parser.add_argument(
        "--matr-output",
        type=Path,
        default=INTERMEDIATE_DIR / "features_matr_sop12.csv",
    )
    parser.add_argument(
        "--hust-output",
        type=Path,
        default=INTERMEDIATE_DIR / "features_hust_sop12.csv",
    )
    parser.add_argument(
        "--combined-output",
        type=Path,
        default=INTERMEDIATE_DIR / "features_matr_hust_sop12.csv",
    )
    parser.add_argument("--windows", nargs="+", type=int, default=list(DEFAULT_WINDOWS))
    parser.add_argument(
        "--skip-hust",
        action="store_true",
        help="Build only MATR SOP12 features when the raw HUST directory is unavailable.",
    )
    return parser.parse_args()


def median_q0(qd: np.ndarray) -> float:
    window = qd[1:5]
    finite = window[np.isfinite(window) & (window > 0)]
    if finite.size == 0:
        return float("nan")
    return float(np.median(finite))


def first_crossing_cycle(cycles: np.ndarray, qd: np.ndarray, fraction: float) -> tuple[float, int]:
    q0 = median_q0(qd)
    if not math.isfinite(q0):
        return float("nan"), 1
    valid = np.isfinite(cycles) & np.isfinite(qd) & (qd > 0)
    hits = np.where(valid & (qd <= fraction * q0))[0]
    if hits.size:
        return float(cycles[hits[0]]), 0
    observed = cycles[valid]
    if observed.size == 0:
        return float("nan"), 1
    return float(observed[-1] + 1), 1


def linear_slope(values: np.ndarray, cycle_numbers: np.ndarray | None = None) -> float:
    finite = np.isfinite(values)
    if cycle_numbers is None:
        cycle_numbers = np.arange(len(values), dtype=float)
    else:
        cycle_numbers = cycle_numbers.astype(float)
    finite &= np.isfinite(cycle_numbers)
    x = cycle_numbers[finite]
    y = values[finite]
    if y.size <= 1 or np.allclose(y, y[0]):
        return 0.0
    return float(np.polyfit(x, y, 1)[0])


def slope_ratio(values: np.ndarray, cycle_numbers: np.ndarray) -> float:
    finite = np.isfinite(values) & np.isfinite(cycle_numbers)
    x = cycle_numbers[finite]
    y = values[finite]
    if y.size < 4:
        return float("nan")
    split = y.size // 2
    first_slope = linear_slope(y[:split], x[:split])
    second_slope = linear_slope(y[split:], x[split:])
    if abs(first_slope) < 1e-12:
        return float("nan")
    return float(second_slope / first_slope)


def qdis_features(qd: np.ndarray, n_cycles: int) -> dict[str, float] | None:
    if qd.size < n_cycles or n_cycles < 10:
        return None

    qd_2_to_n = qd[1:n_cycles]
    cycles_2_to_n = np.arange(2, n_cycles + 1, dtype=float)
    finite = np.isfinite(qd_2_to_n)
    q = qd_2_to_n[finite]
    cyc = cycles_2_to_n[finite]
    if q.size < 4:
        return None

    diffs = np.diff(q)
    drops = -diffs
    qdis_2 = float(qd[1]) if np.isfinite(qd[1]) else float("nan")
    qdis_n = float(qd[n_cycles - 1]) if np.isfinite(qd[n_cycles - 1]) else float("nan")
    qdis_cycle10 = float(qd[9]) if np.isfinite(qd[9]) else float("nan")
    retention_ratio = float(qdis_n / qdis_2) if math.isfinite(qdis_2) and abs(qdis_2) > 1e-12 else float("nan")

    return {
        "Qdis_N": qdis_n,
        "delta_Qdis": float(qdis_n - qdis_2) if math.isfinite(qdis_n) and math.isfinite(qdis_2) else float("nan"),
        "retention_ratio": retention_ratio,
        "slope_linear": linear_slope(q, cyc),
        "variance_Qdis": float(np.var(q)),
        "range_Qdis": float(np.max(q) - np.min(q)),
        "max_drop": float(np.max(drops)) if drops.size else 0.0,
        "std_diff": float(np.std(diffs)) if diffs.size else 0.0,
        "skewness_Qdis": float(skew(q, bias=False)) if q.size >= 3 else float("nan"),
        "slope_ratio": slope_ratio(q, cyc),
        "Qdis_cycle10": qdis_cycle10,
        "mean_diff": float(np.mean(diffs)) if diffs.size else 0.0,
    }


def load_matr_batches() -> dict[str, Any]:
    combined: dict[str, Any] = {}
    for path in [RAW_DIR / "batch1.pkl", RAW_DIR / "batch2.pkl", RAW_DIR / "batch3_varcharge.pkl"]:
        if not path.exists():
            continue
        with path.open("rb") as handle:
            batch = pickle.load(handle)
        overlap = set(combined) & set(batch)
        if overlap:
            raise ValueError(f"Duplicate cell IDs across raw batches: {sorted(overlap)}")
        combined.update(batch)
    if not combined:
        raise SystemExit("No MATR raw pickle files found.")
    return combined


def build_matr_features(labels_path: Path, windows: list[int]) -> pd.DataFrame:
    cells = load_matr_batches()
    rows: list[dict[str, float | int | str]] = []
    for cell_id, cell_data in sorted(cells.items()):
        qd = np.asarray(cell_data["summary"]["QD"], dtype=float).ravel()
        for window in windows:
            features = qdis_features(qd, window)
            if features is None:
                continue
            rows.append(
                {
                    "cell_id": cell_id,
                    "dataset_prefix": "matr",
                    "source_batch": cell_id.split("c", 1)[0],
                    "n_cycles": window,
                    **features,
                }
            )

    df = pd.DataFrame(rows)
    if labels_path.exists():
        labels = pd.read_csv(labels_path)
        label_columns = [column for column in labels.columns if column != "dataset_prefix"]
        df = df.merge(labels[label_columns], on="cell_id", how="left")
    df["cycle_life"] = df["eol_85pct_q0_label"]
    return df


def build_hust_features(hust_dir: Path, windows: list[int]) -> pd.DataFrame:
    if not hust_dir.exists():
        raise SystemExit(f"HUST directory not found: {hust_dir}")

    rows: list[dict[str, float | int | str]] = []
    for path in sorted(hust_dir.glob("*.pkl")):
        with path.open("rb") as handle:
            raw = pickle.load(handle)
        raw_cell_id, cell = next(iter(raw.items()))
        cell_id = f"hust_{raw_cell_id}"
        cycle_keys = sorted(int(cycle) for cycle in cell["dq"])
        qd = np.asarray([float(cell["dq"][cycle]) / 1000.0 for cycle in cycle_keys], dtype=float)
        cycles = np.asarray(cycle_keys, dtype=float)
        eol_80, censored_80 = first_crossing_cycle(cycles, qd, 0.80)
        eol_85, censored_85 = first_crossing_cycle(cycles, qd, 0.85)
        q0 = median_q0(qd)

        for window in windows:
            features = qdis_features(qd, window)
            if features is None:
                continue
            rows.append(
                {
                    "cell_id": cell_id,
                    "dataset_prefix": "hust",
                    "source_batch": "hust",
                    "n_cycles": window,
                    "q0": q0,
                    "q0_source": "median_qd_cycles_2_to_5",
                    "eol_80pct_q0_cycle": eol_80 if not math.isnan(eol_80) else float("nan"),
                    "eol_80pct_q0_label": eol_80,
                    "is_censored_80pct_q0": censored_80,
                    "eol_85pct_q0_cycle": eol_85 if not math.isnan(eol_85) else float("nan"),
                    "eol_85pct_q0_label": eol_85,
                    "is_censored_85pct_q0": censored_85,
                    "cycle_life": eol_85,
                    **features,
                }
            )
    return pd.DataFrame(rows)


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Saved {len(df)} rows to {path}")


def main() -> None:
    args = parse_args()
    matr = build_matr_features(args.labels, args.windows)
    write_csv(matr, args.matr_output)
    if args.skip_hust:
        return

    hust = build_hust_features(args.hust_dir, args.windows)
    common_columns = sorted(set(matr.columns) | set(hust.columns))
    write_csv(hust, args.hust_output)
    write_csv(
        pd.concat([matr.reindex(columns=common_columns), hust.reindex(columns=common_columns)], ignore_index=True),
        args.combined_output,
    )


if __name__ == "__main__":
    main()
