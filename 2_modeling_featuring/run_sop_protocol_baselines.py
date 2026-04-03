"""
Run SOP-style within-dataset and cross-dataset baseline experiments.

This script uses the current feature table only as a transition layer while the
project moves toward the new 12-feature SOP pipeline.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from metrics_utils import bootstrap_metric_ci, compute_metrics
from sklearn.linear_model import ElasticNetCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from split_protocol_utils import (
    DEFAULT_SEEDS,
    load_split_json,
)
from xgboost import XGBRegressor


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET = PROJECT_ROOT / "data" / "intermediate" / "features_top8_cycles.csv"
DEFAULT_SPLIT_DIR = PROJECT_ROOT / "splits"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "results" / "results_sop_protocol_baselines.json"

FEATURES = [
    "IR_delta",
    "dQd_slope",
    "Qd_mean",
    "IR_slope",
    "Tavg_mean",
    "IR_mean",
    "Qd_std",
    "IR_std",
]
PRIMARY_WINDOWS = (100, 50)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SOP-style baseline experiments.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--split-dir", type=Path, default=DEFAULT_SPLIT_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    return parser.parse_args()


def load_feature_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"Feature CSV not found: {path}")
    df = pd.read_csv(path)
    df = df.copy()
    df["cycle_life"] = pd.to_numeric(df["cycle_life"], errors="coerce")
    df.dropna(subset=["cell_id", "n_cycles", "cycle_life"], inplace=True)
    df["dataset_prefix"] = df["cell_id"].astype(str).str.extract(r"^(b\d+)")
    if df["dataset_prefix"].isna().any():
        raise SystemExit("Could not infer dataset prefix from every cell_id.")
    return df


def subset_by_cells(df: pd.DataFrame, cell_ids: list[str]) -> pd.DataFrame:
    return df[df["cell_id"].isin(cell_ids)].copy()


def build_model_factories() -> dict[str, object]:
    return {
        "elastic_net": lambda: make_pipeline(
            StandardScaler(),
            ElasticNetCV(
                l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
                cv=5,
                max_iter=50000,
                random_state=42,
            ),
        ),
        "xgboost": lambda: XGBRegressor(
            max_depth=5,
            learning_rate=0.05,
            n_estimators=400,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=2.0,
            reg_alpha=1.0,
            random_state=42,
            n_jobs=-1,
            tree_method="hist",
        ),
    }


def score_model(model, train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
    X_train = train_df[FEATURES].to_numpy()
    y_train = train_df["cycle_life"].to_numpy()
    X_test = test_df[FEATURES].to_numpy()
    y_test = test_df["cycle_life"].to_numpy()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    metrics = compute_metrics(y_test, preds)
    metrics["bootstrap_95_ci"] = bootstrap_metric_ci(y_test, preds)
    metrics["train_rows"] = int(len(train_df))
    metrics["test_rows"] = int(len(test_df))
    return metrics


def aggregate_seed_metrics(seed_results: list[dict]) -> dict:
    if not seed_results:
        return {}

    summary: dict[str, object] = {"num_seeds": len(seed_results), "per_seed": seed_results}
    for metric in ("MAE", "SMAPE", "R2"):
        values = [result[metric] for result in seed_results]
        summary[metric] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        }
        lowers = [result["bootstrap_95_ci"][metric]["lower"] for result in seed_results]
        uppers = [result["bootstrap_95_ci"][metric]["upper"] for result in seed_results]
        summary[f"{metric}_bootstrap_95_ci"] = {
            "lower_mean": float(np.mean(lowers)),
            "upper_mean": float(np.mean(uppers)),
        }
    return summary


def run_experiment(
    df: pd.DataFrame,
    split_dir: Path,
    train_prefix: str,
    test_prefix: str,
    seeds: list[int],
) -> dict:
    factories = build_model_factories()
    results: dict[str, dict] = {}
    for model_name, factory in factories.items():
        results[model_name] = {}
        for n_cycles in PRIMARY_WINDOWS:
            per_seed: list[dict] = []
            for seed in seeds:
                train_split = load_split_json(split_dir / f"{train_prefix}_{seed}.json")
                test_split = load_split_json(split_dir / f"{test_prefix}_{seed}.json")

                train_df = subset_by_cells(
                    df[(df["dataset_prefix"] == train_prefix) & (df["n_cycles"] == n_cycles)],
                    train_split["train"],
                )
                test_df = subset_by_cells(
                    df[(df["dataset_prefix"] == test_prefix) & (df["n_cycles"] == n_cycles)],
                    test_split["test"],
                )
                if train_df.empty or test_df.empty:
                    continue
                per_seed.append(score_model(factory(), train_df, test_df))
            results[model_name][str(n_cycles)] = aggregate_seed_metrics(per_seed)
    return results


def main() -> None:
    args = parse_args()
    feature_df = load_feature_table(args.dataset)
    summary = {
        "protocol_version": "sop_baseline_transition",
        "notes": [
            "Uses SOP JSON splits, seed averaging, and bootstrap confidence intervals.",
            "Still runs on the current 8-feature table until the 12-feature engineering stage is implemented.",
        ],
        "within_dataset": {
            "b1_to_b1": run_experiment(feature_df, args.split_dir, "b1", "b1", args.seeds),
            "b2_to_b2": run_experiment(feature_df, args.split_dir, "b2", "b2", args.seeds),
        },
        "cross_dataset": {
            "b1_to_b2": run_experiment(feature_df, args.split_dir, "b1", "b2", args.seeds),
            "b2_to_b1": run_experiment(feature_df, args.split_dir, "b2", "b1", args.seeds),
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(f"Saved SOP baseline summary to {args.output}")


if __name__ == "__main__":
    main()
