"""
Generate MAPE comparison plots/tables for RF, XGB, CatBoost.

The script trains each model on the preferred feature set (with and without
``Qd_std``) for n_cycles = 25/50/100 and stores the evaluation artifacts
under ``plots/`` while also emitting a JSON summary of the metrics.

Usage:
    python 2_modeling_featuring/plot_mape_vs_cycles.py
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Callable, Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET = PROJECT_ROOT / "features_top8_cycles.csv"
PLOTS_DIR = PROJECT_ROOT / "plots"
SUMMARY_JSON = PROJECT_ROOT / "results_top8_metrics.json"

BASE_FEATURES = [
    "IR_delta",
    "dQd_slope",
    "Qd_mean",
    "IR_slope",
    "Tavg_mean",
    "IR_mean",
    "Qd_std",
    "IR_std",
]

FEATURE_SETS = {
    "with_qd_std": BASE_FEATURES,
    "without_qd_std": [feat for feat in BASE_FEATURES if feat != "Qd_std"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help="CSV containing the engineered features.",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=SUMMARY_JSON,
        help="Path to store the serialized metrics.",
    )
    parser.add_argument(
        "--plots-dir",
        type=Path,
        default=PLOTS_DIR,
        help="Directory for plot outputs.",
    )
    return parser.parse_args()


def load_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.copy()
    df["cycle_life"] = pd.to_numeric(df["cycle_life"], errors="coerce")
    return df


def prepare_df(df: pd.DataFrame, feature_cols: list[str]) -> Tuple[pd.DataFrame, list[str]]:
    cols = [c for c in feature_cols if c in df.columns]
    cleaned = df.dropna(subset=["cycle_life"] + cols)
    return cleaned, cols


def evaluate_model(
    df: pd.DataFrame,
    feature_cols: list[str],
    model_builder: Callable[[], object],
) -> Dict[int, Dict[str, float]]:
    cleaned, cols = prepare_df(df, feature_cols)
    metrics: Dict[int, Dict[str, float]] = {}

    for n_cycles in (25, 50, 100):
        subset = cleaned[cleaned["n_cycles"] == n_cycles]
        if len(subset) < 10:
            continue
        X = subset[cols]
        y = subset["cycle_life"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        model = model_builder()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

        metrics[n_cycles] = {"MAE": mae, "R2": r2, "MAPE": mape}

    return metrics


def build_models() -> Dict[str, Dict[str, object]]:
    rf_params = dict(
        n_estimators=400,
        min_samples_leaf=2,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1,
    )
    xgb_params = dict(
        n_estimators=800,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=2.0,
        reg_alpha=1.0,
        random_state=42,
        n_jobs=-1,
        tree_method="hist",
    )
    cat_params = dict(
        iterations=400,
        learning_rate=0.05,
        depth=6,
        loss_function="MAE",
        random_seed=42,
        verbose=False,
    )
    return {
        "random_forest": {
            "label": "Random Forest",
            "builder": lambda: RandomForestRegressor(**rf_params),
        },
        "xgboost": {
            "label": "XGBoost",
            "builder": lambda: XGBRegressor(**xgb_params),
        },
        "catboost": {
            "label": "CatBoost",
            "builder": lambda: CatBoostRegressor(**cat_params),
        },
    }


def plot_mape(
    plots_dir: Path,
    model_key: str,
    model_label: str,
    results: Dict[str, Dict[int, Dict[str, float]]],
) -> None:
    plots_dir.mkdir(exist_ok=True)
    cycles = [25, 50, 100]
    plt.figure(figsize=(6, 4))

    for feature_key, style in [
        ("with_qd_std", {"label": "Qd_std var", "marker": "o"}),
        ("without_qd_std", {"label": "Qd_std yok", "marker": "s"}),
    ]:
        mape_values = []
        for n in cycles:
            metric = results.get(feature_key, {}).get(n)
            mape_values.append(metric["MAPE"] if metric else math.nan)
        plt.plot(cycles, mape_values, **style)

    plt.title(f"{model_label} - MAPE vs n_cycles")
    plt.xlabel("n_cycles")
    plt.ylabel("MAPE (%)")
    plt.grid(True, alpha=0.3)
    plt.xticks(cycles)
    plt.legend()
    plt.tight_layout()
    out_path = plots_dir / f"mape_{model_key}.png"
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_table(
    plots_dir: Path,
    model_key: str,
    model_label: str,
    summary: Dict[str, Dict[int, Dict[str, float]]],
) -> None:
    rows = []
    for n in (25, 50, 100):
        with_metrics = summary.get("with_qd_std", {}).get(n)
        without_metrics = summary.get("without_qd_std", {}).get(n)
        if with_metrics and without_metrics:
            rows.append(
                [
                    str(n),
                    f"{with_metrics['MAE']:.2f} | {with_metrics['MAPE']:.2f}",
                    f"{without_metrics['MAE']:.2f} | {without_metrics['MAPE']:.2f}",
                ]
            )

    if not rows:
        return

    fig, ax = plt.subplots(figsize=(4.8, 2.6))
    ax.axis("off")
    header = ["Cycles", "Qd_std var (MAE | MAPE)", "Qd_std yok (MAE | MAPE)"]
    table_data = [header] + rows
    table = ax.table(cellText=table_data, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.3)
    ax.set_title(f"{model_label} – MAE / MAPE", pad=12)
    fig.tight_layout()
    out_path = plots_dir / f"table_{model_key}.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def dump_summary(path: Path, summary: Dict[str, Dict[str, Dict[int, Dict[str, float]]]]) -> None:
    serializable: Dict[str, Dict[str, Dict[str, Dict[str, float]]]] = {}
    for model_key, feature_data in summary.items():
        serializable[model_key] = {}
        for feature_key, metrics in feature_data.items():
            serializable[model_key][feature_key] = {
                str(n): {metric: float(value) for metric, value in stats.items()}
                for n, stats in metrics.items()
            }
    with path.open("w", encoding="utf-8") as fp:
        json.dump(serializable, fp, indent=2)
    print(f"Saved summary to {path}")


def main() -> None:
    args = parse_args()
    if not args.dataset.exists():
        raise SystemExit(f"Dataset bulunamadı: {args.dataset}")

    df = load_dataset(args.dataset)
    models = build_models()
    summary: Dict[str, Dict[str, Dict[int, Dict[str, float]]]] = {}

    for model_key, cfg in models.items():
        summary[model_key] = {}
        for feature_key, feature_list in FEATURE_SETS.items():
            metrics = evaluate_model(df, feature_list, cfg["builder"])
            summary[model_key][feature_key] = metrics
            print(f"{cfg['label']} - {feature_key}: {metrics}")
        plot_mape(args.plots_dir, model_key, cfg["label"], summary[model_key])
        plot_table(args.plots_dir, model_key, cfg["label"], summary[model_key])

    dump_summary(args.summary_json, summary)


if __name__ == "__main__":
    main()
