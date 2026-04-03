"""
Train on one batch and test on the other batch without any adaptation.

This script implements the out-of-scope clarification from the project:
no transfer learning, no domain adaptation, and no fine-tuning. The model is
simply trained on dataset A and evaluated directly on dataset B.

Usage:
    python 2_modeling_featuring/evaluate_cross_dataset_generalization.py \
        --dataset data/intermediate/features_top8_cycles.csv \
        --train-batch b1 \
        --test-batch b2
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from metrics_utils import bootstrap_metric_ci, compute_metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNetCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
INTERMEDIATE_DIR = DATA_DIR / "intermediate"
RESULTS_DIR = PROJECT_ROOT / "outputs" / "results"
DEFAULT_DATASET = INTERMEDIATE_DIR / "features_top8_cycles.csv"

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
    "without_qd_std": [feature for feature in BASE_FEATURES if feature != "Qd_std"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train on one battery batch and test on another batch."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help=f"Feature CSV path (default: {DEFAULT_DATASET})",
    )
    parser.add_argument(
        "--train-batch",
        type=str,
        default="b1",
        help="Source batch prefix to train on (for example: b1).",
    )
    parser.add_argument(
        "--test-batch",
        type=str,
        default="b2",
        help="Target batch prefix to test on (for example: b2).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON output path. Default is derived from train/test batch names.",
    )
    return parser.parse_args()


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"Dataset bulunamadi: {path}")
    df = pd.read_csv(path)
    df = df.copy()
    df["cycle_life"] = pd.to_numeric(df["cycle_life"], errors="coerce")
    df.dropna(subset=["cycle_life", "cell_id", "n_cycles"], inplace=True)
    df["batch_prefix"] = df["cell_id"].astype(str).str.extract(r"^(b\d+)")
    if df["batch_prefix"].isna().any():
        raise SystemExit("Bazi satirlarda cell_id icinden batch prefix cikartilamadi.")
    return df

def build_models() -> dict[str, Callable[[], object]]:
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
    enet_params = dict(
        l1_ratio=[0.1, 0.5, 0.9],
        alphas=np.logspace(-4, 0, 20),
        cv=5,
        max_iter=50000,
        tol=1e-3,
        random_state=42,
    )
    return {
        "random_forest": lambda: RandomForestRegressor(**rf_params),
        "xgboost": lambda: XGBRegressor(**xgb_params),
        "catboost": lambda: CatBoostRegressor(**cat_params),
        "elastic_net": lambda: make_pipeline(StandardScaler(), ElasticNetCV(**enet_params)),
    }

def evaluate_direction(
    df: pd.DataFrame,
    train_batch: str,
    test_batch: str,
    models: dict[str, Callable[[], object]],
) -> dict[str, dict[str, dict[int, dict[str, float]]]]:
    summary: dict[str, dict[str, dict[int, dict[str, float]]]] = {}

    for model_name, build_model in models.items():
        summary[model_name] = {}
        for feature_key, feature_cols in FEATURE_SETS.items():
            cycle_results: dict[int, dict[str, float]] = {}
            for n_cycles in (25, 50, 100):
                train_df = df[
                    (df["batch_prefix"] == train_batch) & (df["n_cycles"] == n_cycles)
                ].copy()
                test_df = df[
                    (df["batch_prefix"] == test_batch) & (df["n_cycles"] == n_cycles)
                ].copy()
                if train_df.empty or test_df.empty:
                    continue

                cols = [col for col in feature_cols if col in train_df.columns]
                train_df.dropna(subset=cols + ["cycle_life"], inplace=True)
                test_df.dropna(subset=cols + ["cycle_life"], inplace=True)
                if train_df.empty or test_df.empty:
                    continue

                model = build_model()
                model.fit(train_df[cols], train_df["cycle_life"])
                preds = model.predict(test_df[cols])

                y_true = test_df["cycle_life"].to_numpy()
                metrics = compute_metrics(y_true, preds)
                metrics["bootstrap_95_ci"] = bootstrap_metric_ci(y_true, preds)
                metrics["train_rows"] = int(len(train_df))
                metrics["test_rows"] = int(len(test_df))
                cycle_results[n_cycles] = metrics

            summary[model_name][feature_key] = cycle_results
    return summary


def main() -> None:
    args = parse_args()
    if args.train_batch == args.test_batch:
        raise SystemExit("train-batch ve test-batch ayni olmamali.")

    output_path = args.output
    if output_path is None:
        output_path = (
            RESULTS_DIR
            / f"results_cross_dataset_{args.train_batch}_to_{args.test_batch}.json"
        )

    df = load_dataset(args.dataset)
    available_batches = sorted(df["batch_prefix"].dropna().unique().tolist())
    if args.train_batch not in available_batches:
        raise SystemExit(
            f"Train batch bulunamadi: {args.train_batch}. Mevcut batch'ler: {available_batches}"
        )
    if args.test_batch not in available_batches:
        raise SystemExit(
            f"Test batch bulunamadi: {args.test_batch}. Mevcut batch'ler: {available_batches}"
        )

    summary = {
        "protocol": "train_on_A_test_on_B_without_adaptation",
        "train_batch": args.train_batch,
        "test_batch": args.test_batch,
        "models": evaluate_direction(df, args.train_batch, args.test_batch, build_models()),
    }

    for model_name, feature_sets in summary["models"].items():
        for feature_key, cycles_data in feature_sets.items():
            print(f"{model_name} ({feature_key}) -> {cycles_data}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(f"Saved cross-dataset results to {output_path}")


if __name__ == "__main__":
    main()
