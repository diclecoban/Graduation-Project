"""
Top-k feature sweep for the SOPv2 battery-life pipeline.

The sweep ranks features from the existing SHAP attribution table, creates
feature-list files for k in {3, 6, 10, 12}, runs the standard within- and
cross-dataset experiments for each feature subset, and writes compact summary
tables for quick advisor-facing comparison.

Inputs:
    data/intermediate/shap_feature_importance.csv
    data/intermediate/features_sop12_combined.csv
    splits/sop_v2/{matr,hust}_{seed}.json

Outputs:
    data/intermediate/feature_set_top{k}_shap.txt
    outputs/results_v2_topk_shap/topk_ranking.csv
    outputs/results_v2_topk_shap/topk_within_best.csv
    outputs/results_v2_topk_shap/topk_cross_best.csv
    outputs/results_v2_topk_shap/k{k}_{within,cross}/...

Usage:
    python3 3_analysis/top_k_sweep.py
    python3 3_analysis/top_k_sweep.py --models catboost random_forest gaussian_process
    python3 3_analysis/top_k_sweep.py --ks 3 6 10 12 --windows 100
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SHAP_PATH = PROJECT_ROOT / "data" / "intermediate" / "shap_feature_importance.csv"
INTERMEDIATE_DIR = PROJECT_ROOT / "data" / "intermediate"
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "results_v2_topk_shap"
RUN_EXPERIMENTS = PROJECT_ROOT / "2_models" / "run_experiments.py"

DEFAULT_KS = [3, 6, 10, 12]
DEFAULT_WINDOWS = [100]
DEFAULT_MODELS = ["catboost", "random_forest", "gaussian_process", "elastic_net"]


def rank_features(shap_path: Path) -> pd.DataFrame:
    shap = pd.read_csv(shap_path)
    primary = shap[shap["n_cycles"] == 100].copy()
    if primary.empty:
        raise ValueError("No N=100 rows found in SHAP table.")

    ranking = (
        primary
        .groupby("feature", as_index=False)
        .agg(
            mean_relative_importance=("relative_importance_mean", "mean"),
            mean_rank=("rank_mean", "mean"),
            top5_rate=("top5_rate", "mean"),
            top10_rate=("top10_rate", "mean"),
        )
        .sort_values(
            ["mean_relative_importance", "top5_rate", "top10_rate", "mean_rank"],
            ascending=[False, False, False, True],
        )
        .reset_index(drop=True)
    )
    ranking.insert(0, "overall_rank", range(1, len(ranking) + 1))
    return ranking


def write_feature_sets(ranking: pd.DataFrame, ks: list[int]) -> dict[int, Path]:
    feature_paths: dict[int, Path] = {}
    features = ranking["feature"].tolist()
    for k in ks:
        if k < 1 or k > len(features):
            raise ValueError(f"k={k} is outside available feature count 1..{len(features)}")
        path = INTERMEDIATE_DIR / f"feature_set_top{k}_shap.txt"
        path.write_text("\n".join(features[:k]) + "\n")
        feature_paths[k] = path
    return feature_paths


def run_experiment(
    feature_path: Path,
    out_dir: Path,
    *,
    models: list[str],
    windows: list[int],
    cross_dataset: bool,
) -> None:
    cmd = [
        sys.executable,
        str(RUN_EXPERIMENTS),
        "--features-from",
        str(feature_path),
        "--output-dir",
        str(out_dir),
        "--models",
        *models,
        "--windows",
        *(str(w) for w in windows),
    ]
    if cross_dataset:
        cmd.append("--cross-dataset")
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)


def summarize_result(summary_path: Path, k: int, mode: str) -> pd.DataFrame:
    df = pd.read_csv(summary_path)
    if df.empty:
        return pd.DataFrame()
    group_cols = ["experiment", "n_cycles"]
    rows = []
    for keys, part in df.groupby(group_cols):
        best_r2 = part.sort_values("R2_mean", ascending=False).iloc[0]
        best_mae = part.sort_values("MAE_mean", ascending=True).iloc[0]
        experiment, n_cycles = keys
        rows.append({
            "k": k,
            "mode": mode,
            "experiment": experiment,
            "n_cycles": int(n_cycles),
            "best_R2_model": best_r2["model"],
            "best_R2": best_r2["R2_mean"],
            "best_R2_MAE": best_r2["MAE_mean"],
            "best_MAE_model": best_mae["model"],
            "best_MAE": best_mae["MAE_mean"],
            "best_MAE_R2": best_mae["R2_mean"],
        })
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ks", type=int, nargs="+", default=DEFAULT_KS)
    parser.add_argument("--windows", type=int, nargs="+", default=DEFAULT_WINDOWS)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--skip-run", action="store_true",
                        help="Only rebuild feature files and summaries from existing result dirs.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not SHAP_PATH.exists():
        print(f"[error] missing {SHAP_PATH.relative_to(PROJECT_ROOT)}")
        return 1

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    ranking = rank_features(SHAP_PATH)
    ranking_path = OUTPUT_ROOT / "topk_ranking.csv"
    ranking.to_csv(ranking_path, index=False)
    print(f"[save] {ranking_path.relative_to(PROJECT_ROOT)}")

    feature_paths = write_feature_sets(ranking, args.ks)
    within_summaries = []
    cross_summaries = []

    for k in args.ks:
        print(f"\n========== top-{k} ==========")
        feature_path = feature_paths[k]
        within_dir = OUTPUT_ROOT / f"k{k}_within"
        cross_dir = OUTPUT_ROOT / f"k{k}_cross"

        if not args.skip_run:
            run_experiment(feature_path, within_dir, models=args.models, windows=args.windows, cross_dataset=False)
            run_experiment(feature_path, cross_dir, models=args.models, windows=args.windows, cross_dataset=True)

        within_summary = within_dir / "results_summary.csv"
        cross_summary = cross_dir / "results_summary.csv"
        if within_summary.exists():
            within_summaries.append(summarize_result(within_summary, k, "within"))
        if cross_summary.exists():
            cross_summaries.append(summarize_result(cross_summary, k, "cross"))

    if within_summaries:
        out = pd.concat(within_summaries, ignore_index=True)
        path = OUTPUT_ROOT / "topk_within_best.csv"
        out.to_csv(path, index=False)
        print(f"[save] {path.relative_to(PROJECT_ROOT)}")
        print(out.to_string(index=False))

    if cross_summaries:
        out = pd.concat(cross_summaries, ignore_index=True)
        path = OUTPUT_ROOT / "topk_cross_best.csv"
        out.to_csv(path, index=False)
        print(f"[save] {path.relative_to(PROJECT_ROOT)}")
        print(out.to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
