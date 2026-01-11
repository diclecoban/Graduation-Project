"""
Select CatBoost hyperparameters with grouped CV, then train on train+val and test.

Usage:
    python 2_modeling_featuring/train_catboost_cv_selected.py \
        --train features_top8_cycles_train.csv \
        --val features_top8_cycles_val.csv \
        --test features_top8_cycles_test.csv
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_TRAIN = PROJECT_ROOT / "features_top8_cycles_train.csv"
DEFAULT_VAL = PROJECT_ROOT / "features_top8_cycles_val.csv"
DEFAULT_TEST = PROJECT_ROOT / "features_top8_cycles_test.csv"
DEFAULT_JSON = PROJECT_ROOT / "results_catboost_cv_selected.json"
DEFAULT_TABLE = PROJECT_ROOT / "plots/table_catboost_cv_selected.png"

CYCLES = (25, 50, 100)
FEATURE_SETS = {
    "with_qd_std": [
        "IR_delta",
        "dQd_slope",
        "Qd_mean",
        "IR_slope",
        "Tavg_mean",
        "IR_mean",
        "Qd_std",
        "IR_std",
    ],
    "without_qd_std": [
        "IR_delta",
        "dQd_slope",
        "Qd_mean",
        "IR_slope",
        "Tavg_mean",
        "IR_mean",
        "IR_std",
    ],
}

CATBOOST_GRID = [
    {"depth": 4, "learning_rate": 0.03, "iterations": 400, "l2_leaf_reg": 3.0},
    {"depth": 4, "learning_rate": 0.05, "iterations": 400, "l2_leaf_reg": 5.0},
    {"depth": 6, "learning_rate": 0.03, "iterations": 600, "l2_leaf_reg": 3.0},
    {"depth": 6, "learning_rate": 0.05, "iterations": 600, "l2_leaf_reg": 5.0},
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tune CatBoost hyperparameters with GroupKFold CV, then score on test."
    )
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN, help="Train CSV path.")
    parser.add_argument("--val", type=Path, default=DEFAULT_VAL, help="Validation CSV path.")
    parser.add_argument("--test", type=Path, default=DEFAULT_TEST, help="Test CSV path.")
    parser.add_argument("--n-splits", type=int, default=5, help="Number of GroupKFold splits.")
    parser.add_argument(
        "--output-json",
        type=Path,
        default=DEFAULT_JSON,
        help="Where to store JSON summary.",
    )
    parser.add_argument(
        "--output-table",
        type=Path,
        default=DEFAULT_TABLE,
        help="Where to store the summary table PNG.",
    )
    return parser.parse_args()


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"CSV not found: {path}")
    df = pd.read_csv(path)
    df = df.copy()
    df["cycle_life"] = pd.to_numeric(df["cycle_life"], errors="coerce")
    return df.dropna(subset=["cycle_life"])


def symmetric_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    diff = np.abs(y_pred - y_true)
    with np.errstate(divide="ignore", invalid="ignore"):
        smape = np.where(denom == 0, 0.0, diff / denom)
    return float(np.mean(smape) * 100.0)


def run_group_cv(
    df: pd.DataFrame,
    feature_cols: list[str],
    params: dict,
    n_splits: int,
) -> dict:
    groups = df["cell_id"].values
    if len(np.unique(groups)) < n_splits:
        raise ValueError("Not enough unique cells for the requested number of splits.")

    X = df[feature_cols].to_numpy()
    y = df["cycle_life"].to_numpy()

    metrics: List[dict] = []
    gkf = GroupKFold(n_splits=n_splits)
    for train_idx, val_idx in gkf.split(X, y, groups):
        model = CatBoostRegressor(
            depth=params["depth"],
            learning_rate=params["learning_rate"],
            iterations=params["iterations"],
            l2_leaf_reg=params["l2_leaf_reg"],
            loss_function="MAE",
            random_seed=42,
            verbose=False,
        )
        model.fit(X[train_idx], y[train_idx])
        preds = model.predict(X[val_idx])
        metrics.append(
            {
                "MAE": float(mean_absolute_error(y[val_idx], preds)),
                "R2": float(r2_score(y[val_idx], preds)),
                "MAPE": float(np.mean(np.abs((y[val_idx] - preds) / y[val_idx])) * 100.0),
                "SMAPE": symmetric_mape(y[val_idx], preds),
            }
        )

    summary = {}
    for metric in ["MAE", "R2", "MAPE", "SMAPE"]:
        values = [fold[metric] for fold in metrics]
        summary[metric] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        }
    return summary


def select_best_config(
    df: pd.DataFrame,
    feature_cols: list[str],
    n_splits: int,
) -> tuple[dict, dict]:
    best = None
    best_summary = {}
    for params in CATBOOST_GRID:
        summary = run_group_cv(df, feature_cols, params, n_splits)
        mae_mean = summary["MAE"]["mean"]
        if best is None or mae_mean < best:
            best = mae_mean
            best_summary = summary | {"params": params}
    return best_summary["params"], best_summary


def train_and_test(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    params: dict,
) -> dict:
    model = CatBoostRegressor(
        depth=params["depth"],
        learning_rate=params["learning_rate"],
        iterations=params["iterations"],
        l2_leaf_reg=params["l2_leaf_reg"],
        loss_function="MAE",
        random_seed=42,
        verbose=False,
    )
    model.fit(train_df[feature_cols], train_df["cycle_life"])
    preds = model.predict(test_df[feature_cols])
    return {
        "MAE": float(mean_absolute_error(test_df["cycle_life"], preds)),
        "R2": float(r2_score(test_df["cycle_life"], preds)),
        "MAPE": float(np.mean(np.abs((test_df["cycle_life"] - preds) / test_df["cycle_life"])) * 100.0),
        "SMAPE": symmetric_mape(test_df["cycle_life"], preds),
    }


def render_table(summary: dict, output_path: Path) -> None:
    rows = []
    for feature_key, cycles_data in summary.items():
        for n_cycle, data in cycles_data.items():
            params = data["best_params"]
            cv_mae = data["cv_metrics"]["MAE"]
            test_metrics = data["test_metrics"]
            rows.append(
                [
                    feature_key,
                    str(n_cycle),
                    params["depth"],
                    params["learning_rate"],
                    params["iterations"],
                    f"{cv_mae['mean']:.2f} ± {cv_mae['std']:.2f}",
                    f"{test_metrics['MAE']:.2f}",
                    f"{test_metrics['R2']:.2f}",
                    f"{test_metrics['MAPE']:.2f}",
                    f"{test_metrics['SMAPE']:.2f}",
                ]
            )

    if not rows:
        print("No rows to plot for CatBoost CV selection table.")
        return

    PLOTS_DIR = output_path.parent
    PLOTS_DIR.mkdir(exist_ok=True)
    header = [
        "Feature set",
        "n_cycles",
        "depth",
        "lr",
        "iters",
        "CV MAE (mean ± std)",
        "Test MAE",
        "Test R2",
        "Test MAPE",
        "Test SMAPE",
    ]
    table_data = [header] + rows
    fig_height = max(2.5, 0.35 * len(rows) + 1.0)
    fig, ax = plt.subplots(figsize=(10.5, fig_height))
    ax.axis("off")
    table = ax.table(cellText=table_data, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.1)
    ax.set_title("CatBoost – CV-selected hyperparameters & test metrics", pad=12)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"Saved {output_path}")


def main() -> None:
    args = parse_args()
    train_df = load_csv(args.train)
    val_df = load_csv(args.val)
    test_df = load_csv(args.test)

    train_full = pd.concat([train_df, val_df], ignore_index=True)

    summary: Dict[str, dict] = {}
    for feature_key, feature_cols in FEATURE_SETS.items():
        summary[feature_key] = {}
        for n_cycles in CYCLES:
            train_subset = train_full[train_full["n_cycles"] == n_cycles]
            test_subset = test_df[test_df["n_cycles"] == n_cycles]
            if train_subset.empty or test_subset.empty:
                continue
            try:
                best_params, cv_metrics = select_best_config(
                    train_subset, feature_cols, args.n_splits
                )
            except ValueError as exc:
                print(f"Skipping {feature_key} @ {n_cycles} cycles: {exc}")
                continue
            test_metrics = train_and_test(train_subset, test_subset, feature_cols, best_params)
            summary[feature_key][n_cycles] = {
                "best_params": best_params,
                "cv_metrics": cv_metrics,
                "test_metrics": test_metrics,
            }
            print(
                f"{feature_key} n={n_cycles} best={best_params} "
                f"CV MAE={cv_metrics['MAE']['mean']:.2f} "
                f"Test MAE={test_metrics['MAE']:.2f}"
            )

    with args.output_json.open("w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2)
    print(f"Saved JSON summary to {args.output_json}")
    render_table(summary, args.output_table)


if __name__ == "__main__":
    main()
