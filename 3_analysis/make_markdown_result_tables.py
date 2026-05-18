"""
Create pipe-style Markdown/text tables from the key advisor CSV results.

Inputs:
    outputs/results_v2_topk_shap/topk_cross_best.csv
    outputs/results_v2_coral_target_calibration/results_summary.csv

Outputs:
    outputs/advisor_tables/topk_cross_best_table.md
    outputs/advisor_tables/coral_target_calibration_table.md
    outputs/advisor_tables/email_ready_result_tables.md

Usage:
    python3 3_analysis/make_markdown_result_tables.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TOPK_PATH = PROJECT_ROOT / "outputs" / "results_v2_topk_shap" / "topk_cross_best.csv"
CORAL_PATH = PROJECT_ROOT / "outputs" / "results_v2_coral_target_calibration" / "results_summary.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "advisor_tables"


def markdown_table(df: pd.DataFrame) -> str:
    headers = list(df.columns)
    rows = [[str(value) for value in row] for row in df.to_numpy()]
    widths = []
    for i, header in enumerate(headers):
        max_row = max((len(row[i]) for row in rows), default=0)
        widths.append(max(len(header), max_row))

    def fmt_row(values: list[str]) -> str:
        return "| " + " | ".join(value.ljust(widths[i]) for i, value in enumerate(values)) + " |"

    sep = "| " + " | ".join("-" * widths[i] for i in range(len(widths))) + " |"
    lines = [fmt_row(headers), sep]
    lines.extend(fmt_row(row) for row in rows)
    return "\n".join(lines)


def topk_table() -> pd.DataFrame:
    df = pd.read_csv(TOPK_PATH)
    out = pd.DataFrame({
        "k": df["k"].astype(int),
        "Direction": df["experiment"].map({
            "hust_to_matr": "HUST -> MATR",
            "matr_to_hust": "MATR -> HUST",
        }),
        "Best model": df["best_R2_model"],
        "Best R2": df["best_R2"].map(lambda x: f"{x:.3f}"),
        "MAE": df["best_R2_MAE"].map(lambda x: f"{x:.1f}"),
    })
    return out


def coral_table() -> pd.DataFrame:
    df = pd.read_csv(CORAL_PATH)
    rows = []
    for experiment in ["hust_to_matr", "matr_to_hust"]:
        part = df[df["experiment"] == experiment]
        direction = "HUST -> MATR" if experiment == "hust_to_matr" else "MATR -> HUST"
        for _, row in part.iterrows():
            setting = "CORAL only" if row["adapter"] == "none" else "CORAL + target calibration"
            rows.append({
                "Direction": direction,
                "Setting": setting,
                "k target": int(row["k_target"]),
                "R2": f"{row['R2_mean']:.3f}",
                "MAE": f"{row['MAE_mean']:.1f}",
                "sMAPE": f"{row['SMAPE_mean']:.2f}",
                "Runs": int(row["n_runs"]),
            })
    return pd.DataFrame(rows)


def main() -> int:
    if not TOPK_PATH.exists():
        print(f"[error] missing {TOPK_PATH.relative_to(PROJECT_ROOT)}")
        return 1
    if not CORAL_PATH.exists():
        print(f"[error] missing {CORAL_PATH.relative_to(PROJECT_ROOT)}")
        return 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    topk_md = markdown_table(topk_table())
    coral_md = markdown_table(coral_table())

    topk_path = OUTPUT_DIR / "topk_cross_best_table.md"
    coral_path = OUTPUT_DIR / "coral_target_calibration_table.md"
    combined_path = OUTPUT_DIR / "email_ready_result_tables.md"

    topk_path.write_text(topk_md + "\n")
    coral_path.write_text(coral_md + "\n")
    combined_path.write_text(
        "Top-k Cross-Dataset Results\n\n"
        f"{topk_md}\n\n"
        "CORAL + Target-Side Calibration Results\n\n"
        f"{coral_md}\n"
    )

    print(f"[save] {topk_path.relative_to(PROJECT_ROOT)}")
    print(f"[save] {coral_path.relative_to(PROJECT_ROOT)}")
    print(f"[save] {combined_path.relative_to(PROJECT_ROOT)}")
    print("\nTop-k Cross-Dataset Results\n")
    print(topk_md)
    print("\nCORAL + Target-Side Calibration Results\n")
    print(coral_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
