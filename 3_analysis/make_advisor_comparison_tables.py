"""
Build the two advisor-requested comparison tables.

Tables:
  1. Point prediction / transfer comparison:
     all features, top-k, raw transfer, CORAL-only, target calibration,
     and CORAL + target calibration.
  2. Conformal prediction comparison:
     source CP, CORAL-after-source CP, target-domain CP, and target-adapted CP.

Inputs are the existing experiment CSVs under outputs/.

Usage:
    python3 3_analysis/make_advisor_comparison_tables.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "outputs" / "advisor_tables"

WITHIN_PATH = PROJECT_ROOT / "outputs" / "results_v2_34feat_log" / "results_summary.csv"
TOPK_WITHIN_PATH = PROJECT_ROOT / "outputs" / "results_v2_topk_shap" / "topk_within_best.csv"
TOPK_CROSS_PATH = PROJECT_ROOT / "outputs" / "results_v2_topk_shap" / "topk_cross_best.csv"
TARGET_CAL_PATH = PROJECT_ROOT / "outputs" / "results_v2_target_rescale" / "results_summary.csv"
CORAL_PATH = PROJECT_ROOT / "outputs" / "results_v2_coral_target_calibration" / "results_summary.csv"
CP_PATH = PROJECT_ROOT / "outputs" / "results_v2_conformal" / "paper_cp_summary.csv"
CORAL_CP_PATH = PROJECT_ROOT / "outputs" / "results_v2_coral_source_cp" / "results_summary.csv"


def fmt_num(value: float, digits: int = 3) -> str:
    if pd.isna(value):
        return ""
    return f"{float(value):.{digits}f}"


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    headers = [str(c) for c in df.columns]
    rows = [[str(v) for v in row] for row in df.to_numpy()]
    widths = [max([len(headers[i])] + [len(row[i]) for row in rows]) for i in range(len(headers))]
    header = "| " + " | ".join(headers[i].ljust(widths[i]) for i in range(len(headers))) + " |"
    sep = "| " + " | ".join("-" * widths[i] for i in range(len(headers))) + " |"
    body = ["| " + " | ".join(row[i].ljust(widths[i]) for i in range(len(headers))) + " |" for row in rows]
    return "\n".join([header, sep, *body])


def require(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing required input: {path.relative_to(PROJECT_ROOT)}")


def direction_label(experiment: str) -> str:
    return experiment.replace("_to_", " -> ").upper()


def best_full_within(within: pd.DataFrame) -> pd.DataFrame:
    rows = []
    part = within[within["n_cycles"].eq(100)].copy()
    for experiment, group in part[part["experiment"].isin(["hust_to_hust", "matr_to_matr"])].groupby("experiment"):
        best = group.sort_values("R2_mean", ascending=False).iloc[0]
        rows.append({
            "Direction": direction_label(experiment),
            "Setting": "All features, within-dataset",
            "Model": best["model"],
            "k": "",
            "R2": best["R2_mean"],
            "MAE": best["MAE_mean"],
            "Note": "Full feature reference",
        })
    return pd.DataFrame(rows)


def best_topk_within(topk_within: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for experiment, group in topk_within.groupby("experiment"):
        best = group.sort_values("best_R2", ascending=False).iloc[0]
        rows.append({
            "Direction": direction_label(experiment),
            "Setting": "Best top-k, within-dataset",
            "Model": best["best_R2_model"],
            "k": int(best["k"]),
            "R2": best["best_R2"],
            "MAE": best["best_R2_MAE"],
            "Note": "Compact feature subset",
        })
    return pd.DataFrame(rows)


def best_topk_cross(topk_cross: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for experiment, group in topk_cross.groupby("experiment"):
        best = group.sort_values("best_R2", ascending=False).iloc[0]
        rows.append({
            "Direction": direction_label(experiment),
            "Setting": "Best top-k raw transfer",
            "Model": best["best_R2_model"],
            "k": int(best["k"]),
            "R2": best["best_R2"],
            "MAE": best["best_R2_MAE"],
            "Note": "Feature selection only",
        })
    return pd.DataFrame(rows)


def best_raw_transfer(target_cal: pd.DataFrame) -> pd.DataFrame:
    rows = []
    part = target_cal[target_cal["n_cycles"].eq(100)].copy()
    for experiment, group in part.groupby("experiment"):
        best = group.sort_values("baseline_R2", ascending=False).iloc[0]
        rows.append({
            "Direction": direction_label(experiment),
            "Setting": "Raw transfer, all features",
            "Model": best["model"],
            "k": "",
            "R2": best["baseline_R2"],
            "MAE": best["baseline_MAE"],
            "Note": "No target labels",
        })
    return pd.DataFrame(rows)


def best_target_calibration(target_cal: pd.DataFrame, k: int = 20) -> pd.DataFrame:
    rows = []
    r2_col = f"k{k}_R2"
    mae_col = f"k{k}_MAE"
    part = target_cal[target_cal["n_cycles"].eq(100)].copy()
    for experiment, group in part.groupby("experiment"):
        best = group.sort_values(r2_col, ascending=False).iloc[0]
        rows.append({
            "Direction": direction_label(experiment),
            "Setting": "Target calibration, all features",
            "Model": best["model"],
            "k": k,
            "R2": best[r2_col],
            "MAE": best[mae_col],
            "Note": "Linear target-side calibration",
        })
    return pd.DataFrame(rows)


def coral_rows(coral: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in coral[coral["n_cycles"].eq(100)].iterrows():
        setting = "CORAL-only" if row["adapter"] == "none" else "CORAL + target calibration"
        rows.append({
            "Direction": direction_label(row["experiment"]),
            "Setting": setting,
            "Model": "coral_mlp",
            "k": int(row["k_target"]),
            "R2": row["R2_mean"],
            "MAE": row["MAE_mean"],
            "Note": "Covariate alignment baseline" if row["adapter"] == "none" else "CORAL plus residual target shift",
        })
    return pd.DataFrame(rows)


def build_point_table() -> pd.DataFrame:
    for path in [WITHIN_PATH, TOPK_WITHIN_PATH, TOPK_CROSS_PATH, TARGET_CAL_PATH, CORAL_PATH]:
        require(path)
    within = pd.read_csv(WITHIN_PATH)
    topk_within = pd.read_csv(TOPK_WITHIN_PATH)
    topk_cross = pd.read_csv(TOPK_CROSS_PATH)
    target_cal = pd.read_csv(TARGET_CAL_PATH)
    coral = pd.read_csv(CORAL_PATH)

    table = pd.concat(
        [
            best_full_within(within),
            best_topk_within(topk_within),
            best_raw_transfer(target_cal),
            best_topk_cross(topk_cross),
            coral_rows(coral),
            best_target_calibration(target_cal, k=20),
        ],
        ignore_index=True,
    )
    order = {
        "All features, within-dataset": 0,
        "Best top-k, within-dataset": 1,
        "Raw transfer, all features": 2,
        "Best top-k raw transfer": 3,
        "CORAL-only": 4,
        "Target calibration, all features": 5,
        "CORAL + target calibration": 6,
    }
    table["order"] = table["Setting"].map(order)
    table = table.sort_values(["Direction", "order"]).drop(columns=["order"]).reset_index(drop=True)
    table["R2"] = table["R2"].map(lambda x: fmt_num(x, 3))
    table["MAE"] = table["MAE"].map(lambda x: fmt_num(x, 1))
    return table


def build_cp_table(confidence_level: float = 0.90) -> pd.DataFrame:
    for path in [CP_PATH, CORAL_CP_PATH]:
        require(path)
    cp = pd.read_csv(CP_PATH)
    coral_cp = pd.read_csv(CORAL_CP_PATH)
    cp = cp[cp["confidence_level"].round(6).eq(round(confidence_level, 6))].copy()
    coral_cp = coral_cp[coral_cp["confidence_level"].round(6).eq(round(confidence_level, 6))].copy()

    rows = []
    labels = {
        "cross_source_calibrated_cp": "Source CP",
        "cross_target_calibrated_cp": "Target-domain CP",
        "cross_target_adapted_cp": "Target-adapted CP",
    }
    keep = cp[cp["scenario"].isin(labels)].copy()
    for _, row in keep.iterrows():
        rows.append({
            "Direction": f"{str(row['source']).upper()} -> {str(row['target']).upper()}",
            "Setting": labels[row["scenario"]],
            "Model": row["model"],
            "Confidence": confidence_level,
            "k target": "" if pd.isna(row["k_target"]) else int(row["k_target"]),
            "k adapter": "" if pd.isna(row["k_adapter"]) else int(row["k_adapter"]),
            "Coverage": row["coverage_mean"],
            "Median width": row["median_width_mean"],
            "R2": row["R2_mean"],
            "Runs": int(row["n_runs"]),
        })
    for _, row in coral_cp.iterrows():
        rows.append({
            "Direction": direction_label(row["experiment"]),
            "Setting": "CORAL-after-source CP",
            "Model": "coral_mlp",
            "Confidence": confidence_level,
            "k target": 0,
            "k adapter": "",
            "Coverage": row["coverage_mean"],
            "Median width": row["median_width_mean"],
            "R2": row["R2_mean"],
            "Runs": int(row["n_runs"]),
        })

    table = pd.DataFrame(rows)
    order = {
        "Source CP": 0,
        "CORAL-after-source CP": 1,
        "Target-domain CP": 2,
        "Target-adapted CP": 3,
    }
    table["order"] = table["Setting"].map(order)
    table = table.sort_values(["Direction", "order", "Model"]).drop(columns=["order"]).reset_index(drop=True)
    for col in ["Confidence", "Coverage", "R2"]:
        table[col] = table[col].map(lambda x: fmt_num(x, 3))
    table["Median width"] = table["Median width"].map(lambda x: fmt_num(x, 1))
    return table


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    point = build_point_table()
    cp = build_cp_table(confidence_level=0.90)

    point_csv = OUT_DIR / "advisor_point_comparison.csv"
    point_md = OUT_DIR / "advisor_point_comparison.md"
    cp_csv = OUT_DIR / "advisor_conformal_comparison.csv"
    cp_md = OUT_DIR / "advisor_conformal_comparison.md"
    narrative = OUT_DIR / "advisor_followup_narrative.md"

    point.to_csv(point_csv, index=False)
    cp.to_csv(cp_csv, index=False)
    point_md.write_text("# Advisor Point-Prediction Comparison\n\n" + dataframe_to_markdown(point) + "\n")
    cp_md.write_text("# Advisor Conformal-Prediction Comparison\n\n" + dataframe_to_markdown(cp) + "\n")
    narrative.write_text(
        "# Advisor Follow-up Narrative\n\n"
        "Top-k feature selection gives a compact within-dataset representation, especially for HUST, but it does not solve cross-dataset transfer: the best top-k raw-transfer rows remain negative-R2.\n\n"
        "The shift should be described as concept shift rather than only a target-scale difference. The 2.09x lifetime ratio is useful evidence of a central scale mismatch, but the stronger point is that the same early-life features do not preserve the same feature-to-lifetime relationship across MATR and HUST.\n\n"
        "CORAL is presented as a covariate-alignment baseline. CORAL-only remains negative-R2, and CORAL-after-source CP still does not provide reliable target-domain uncertainty. This supports the claim that aligning marginal feature distributions is insufficient.\n\n"
        "Target-side calibration and target-domain conformal prediction are the stronger controls under concept shift. A small labeled target set materially improves point prediction, and target-domain/adapted CP restores coverage much more effectively than source-calibrated CP.\n"
    )

    print(f"[save] {point_csv.relative_to(PROJECT_ROOT)}")
    print(f"[save] {point_md.relative_to(PROJECT_ROOT)}")
    print(f"[save] {cp_csv.relative_to(PROJECT_ROOT)}")
    print(f"[save] {cp_md.relative_to(PROJECT_ROOT)}")
    print(f"[save] {narrative.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
