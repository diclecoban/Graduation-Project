"""
CORAL-aligned source conformal prediction baseline.

This script adds the advisor-requested "after CORAL source CP" control:

    1. Train the CORAL representation model on source train labels plus
       unlabeled target features.
    2. Calibrate split-CP residuals on source calibration cells.
    3. Evaluate intervals on the full target dataset.

The goal is diagnostic, not a proposed final solution. It tests whether
covariate alignment plus source-domain calibration is enough under concept
shift.

Outputs:
    outputs/results_v2_coral_source_cp/results_detailed.csv
    outputs/results_v2_coral_source_cp/results_summary.csv
    outputs/results_v2_coral_source_cp/results_coral_source_cp.json

Usage:
    python3 3_analysis/coral_source_conformal.py
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
sys.path.insert(0, str(PROJECT_ROOT / "2_models"))
sys.path.insert(0, str(HERE))

from metrics_utils import compute_metrics  # noqa: E402
from run_experiments import META_COLS, SEEDS  # noqa: E402
from coral_target_calibration import (  # noqa: E402
    predict_cycles,
    train_coral_model,
)

FEATURES_PATH = PROJECT_ROOT / "data" / "intermediate" / "features_sop12_combined.csv"
SPLITS_DIR = PROJECT_ROOT / "splits" / "sop_v2"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "results_v2_coral_source_cp"
DEFAULT_CONFIDENCE_LEVELS = [0.90, 0.95]


def load_split(dataset: str, seed: int) -> dict:
    with (SPLITS_DIR / f"{dataset}_{seed}.json").open() as f:
        return json.load(f)


def dataset_window(df: pd.DataFrame, dataset: str, n_cycles: int) -> pd.DataFrame:
    return df[(df["dataset"] == dataset) & (df["n_cycles"] == n_cycles) & (df["is_censored"] == 0)].copy()


def finite_sample_quantile(abs_residuals: np.ndarray, alpha: float) -> tuple[float, bool, int]:
    residuals = np.sort(np.asarray(abs_residuals, dtype=float))
    n = len(residuals)
    if n == 0:
        return float("nan"), False, 0
    rank = int(math.ceil((n + 1) * (1.0 - alpha)))
    if rank > n:
        return float("inf"), False, rank
    return float(residuals[rank - 1]), True, rank


def evaluate_once(
    df: pd.DataFrame,
    feature_cols: list[str],
    *,
    source: str,
    target: str,
    n_cycles: int,
    seed: int,
    confidence_levels: list[float],
    epochs: int,
    lr: float,
    weight_decay: float,
    lambda_coral: float,
    hidden_dim: int,
    latent_dim: int,
    dropout: float,
) -> list[dict]:
    source_df = dataset_window(df, source, n_cycles)
    target_df = dataset_window(df, target, n_cycles)
    split = load_split(source, seed)
    source_train = source_df[source_df["cell_id"].isin(split["train"])].copy()
    source_cal = source_df[source_df["cell_id"].isin(split["calibration"])].copy()

    if len(source_train) < 5 or len(source_cal) < 2 or len(target_df) < 2:
        return []

    x_source = source_train[feature_cols].to_numpy(dtype=float)
    y_source = source_train["cycle_life"].to_numpy(dtype=float)
    x_target = target_df[feature_cols].to_numpy(dtype=float)
    y_target = target_df["cycle_life"].to_numpy(dtype=float)

    model, scaler, y_log_mean, y_log_std = train_coral_model(
        x_source,
        y_source,
        x_target,
        seed=seed,
        epochs=epochs,
        lr=lr,
        weight_decay=weight_decay,
        lambda_coral=lambda_coral,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        dropout=dropout,
    )

    y_cal = source_cal["cycle_life"].to_numpy(dtype=float)
    pred_cal = predict_cycles(
        model,
        scaler,
        source_cal[feature_cols].to_numpy(dtype=float),
        y_log_mean=y_log_mean,
        y_log_std=y_log_std,
    )
    pred_target = predict_cycles(
        model,
        scaler,
        x_target,
        y_log_mean=y_log_mean,
        y_log_std=y_log_std,
    )
    point_metrics = compute_metrics(y_target, pred_target)

    rows = []
    residuals = np.abs(y_cal - pred_cal)
    for confidence_level in confidence_levels:
        alpha = 1.0 - confidence_level
        q_hat, finite_q, quantile_rank = finite_sample_quantile(residuals, alpha)
        lower = pred_target - q_hat
        upper = pred_target + q_hat
        covered = (y_target >= lower) & (y_target <= upper)
        width = upper - lower
        rows.append({
            "scenario": "cross_coral_source_calibrated_cp",
            "source": source,
            "target": target,
            "experiment": f"{source}_to_{target}",
            "calibration_domain": source,
            "n_cycles": int(n_cycles),
            "seed": int(seed),
            "confidence_level": float(confidence_level),
            "q_hat": float(q_hat),
            "finite_q": bool(finite_q),
            "quantile_rank": int(quantile_rank),
            "n_train": int(len(source_train)),
            "n_calibration": int(len(source_cal)),
            "n_test": int(len(target_df)),
            "covered_count": int(np.sum(covered)),
            "coverage": float(np.mean(covered)),
            "mean_width": float(np.mean(width)),
            "median_width": float(np.median(width)),
            "MAE": point_metrics["MAE"],
            "SMAPE": point_metrics["SMAPE"],
            "R2": point_metrics["R2"],
        })
    return rows


def summarize(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return rows
    return (
        rows
        .groupby(["scenario", "experiment", "source", "target", "calibration_domain", "n_cycles", "confidence_level"], as_index=False)
        .agg(
            coverage_mean=("coverage", "mean"),
            coverage_std=("coverage", "std"),
            mean_width_mean=("mean_width", "mean"),
            median_width_mean=("median_width", "mean"),
            MAE_mean=("MAE", "mean"),
            SMAPE_mean=("SMAPE", "mean"),
            R2_mean=("R2", "mean"),
            q_hat_mean=("q_hat", "mean"),
            finite_q_mean=("finite_q", "mean"),
            n_runs=("coverage", "count"),
            n_train_mean=("n_train", "mean"),
            n_calibration_mean=("n_calibration", "mean"),
            n_test_mean=("n_test", "mean"),
        )
        .sort_values(["experiment", "confidence_level"])
        .reset_index(drop=True)
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--features-from", type=Path, default=None)
    parser.add_argument("--n-cycles", type=int, default=100)
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--confidence-levels", type=float, nargs="+", default=DEFAULT_CONFIDENCE_LEVELS)
    parser.add_argument("--epochs", type=int, default=800)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--lambda-coral", type=float, default=1.0)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--latent-dim", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not FEATURES_PATH.exists():
        print(f"[error] missing {FEATURES_PATH.relative_to(PROJECT_ROOT)}")
        return 1

    df = pd.read_csv(FEATURES_PATH)
    available = [c for c in df.columns if c not in META_COLS]
    feature_cols = list(available)
    feature_source = f"auto-detected from CSV ({len(feature_cols)} features)"
    if args.features_from is not None:
        if not args.features_from.exists():
            print(f"[error] --features-from not found: {args.features_from}")
            return 1
        feature_cols = [line.strip() for line in args.features_from.read_text().splitlines() if line.strip()]
        unknown = [c for c in feature_cols if c not in available]
        if unknown:
            print(f"[error] unknown feature columns: {unknown}")
            return 1
        feature_source = f"loaded from {args.features_from} ({len(feature_cols)} features)"

    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict] = []
    for source, target in [("hust", "matr"), ("matr", "hust")]:
        for seed in args.seeds:
            print(f"[run] CORAL source CP {source}->{target} seed={seed}")
            all_rows.extend(evaluate_once(
                df,
                feature_cols,
                source=source,
                target=target,
                n_cycles=args.n_cycles,
                seed=seed,
                confidence_levels=args.confidence_levels,
                epochs=args.epochs,
                lr=args.lr,
                weight_decay=args.weight_decay,
                lambda_coral=args.lambda_coral,
                hidden_dim=args.hidden_dim,
                latent_dim=args.latent_dim,
                dropout=args.dropout,
            ))

    detailed = pd.DataFrame(all_rows)
    summary = summarize(detailed)
    detailed_path = args.output_dir / "results_detailed.csv"
    summary_path = args.output_dir / "results_summary.csv"
    json_path = args.output_dir / "results_coral_source_cp.json"
    detailed.to_csv(detailed_path, index=False)
    summary.to_csv(summary_path, index=False)
    json_path.write_text(json.dumps({
        "protocol": "coral_source_conformal_v1",
        "feature_set": feature_source,
        "feature_columns": feature_cols,
        "confidence_levels": args.confidence_levels,
        "n_cycles": args.n_cycles,
        "seeds": args.seeds,
        "epochs": args.epochs,
        "lambda_coral": args.lambda_coral,
        "summary": summary.to_dict(orient="records"),
    }, indent=2))

    print(f"[save] {detailed_path.relative_to(PROJECT_ROOT)}")
    print(f"[save] {summary_path.relative_to(PROJECT_ROOT)}")
    print(f"[save] {json_path.relative_to(PROJECT_ROOT)}")
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
