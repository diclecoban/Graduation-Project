"""
Create advisor-friendly HTML tables from the key follow-up CSV outputs.

The CSV files are reproducible but hard to scan in an email. This script turns
the two most important result tables into compact, styled HTML files:

1. Top-k cross-dataset sweep: feature selection alone does not fix transfer.
2. CORAL + residual target calibration: small target calibration moves R2 much
   closer to zero and reduces MAE.

Outputs:
    outputs/advisor_tables/topk_cross_best_table.html
    outputs/advisor_tables/coral_target_calibration_table.html
    outputs/advisor_tables/advisor_email_tables_summary.md

Usage:
    python3 3_analysis/make_advisor_result_tables.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TOPK_PATH = PROJECT_ROOT / "outputs" / "results_v2_topk_shap" / "topk_cross_best.csv"
CORAL_PATH = PROJECT_ROOT / "outputs" / "results_v2_coral_target_calibration" / "results_summary.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "advisor_tables"


CSS = """
body {
  margin: 32px;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Arial, sans-serif;
  color: #111827;
  background: #ffffff;
}
.wrap {
  max-width: 980px;
}
h1 {
  font-size: 24px;
  margin: 0 0 8px;
}
p {
  color: #475569;
  font-size: 14px;
  margin: 0 0 18px;
}
table {
  border-collapse: collapse;
  width: 100%;
  font-size: 14px;
  box-shadow: 0 1px 3px rgba(15, 23, 42, 0.08);
}
th {
  background: #1f2937;
  color: #ffffff;
  padding: 11px 10px;
  text-align: center;
  font-weight: 700;
}
td {
  border: 1px solid #d0d7de;
  padding: 10px;
  text-align: center;
}
tr:nth-child(even) td {
  background: #f8fafc;
}
.r2-good {
  background: #dcfce7 !important;
  font-weight: 700;
}
.r2-mid {
  background: #fef9c3 !important;
  font-weight: 700;
}
.r2-bad {
  background: #fee2e2 !important;
  font-weight: 700;
}
.note {
  margin-top: 14px;
  font-size: 13px;
  color: #64748b;
}
"""


def r2_class(value: float) -> str:
    if value > -0.5:
        return "r2-good"
    if value > -2.0:
        return "r2-mid"
    return "r2-bad"


def table_html(df: pd.DataFrame, *, r2_columns: list[str]) -> str:
    lines = ["<table>", "<thead><tr>"]
    for col in df.columns:
        lines.append(f"<th>{col}</th>")
    lines.append("</tr></thead>")
    lines.append("<tbody>")
    for _, row in df.iterrows():
        lines.append("<tr>")
        for col in df.columns:
            value = row[col]
            klass = ""
            if col in r2_columns:
                klass = f' class="{r2_class(float(value))}"'
                value = f"{float(value):.3f}"
            elif isinstance(value, float):
                value = f"{value:.1f}"
            lines.append(f"<td{klass}>{value}</td>")
        lines.append("</tr>")
    lines.append("</tbody></table>")
    return "\n".join(lines)


def write_html(path: Path, *, title: str, subtitle: str, df: pd.DataFrame, r2_columns: list[str], note: str) -> None:
    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>{CSS}</style>
</head>
<body>
<main class="wrap">
<h1>{title}</h1>
<p>{subtitle}</p>
{table_html(df, r2_columns=r2_columns)}
<div class="note">{note}</div>
</main>
</body>
</html>
"""
    path.write_text(html)


def make_topk_table() -> pd.DataFrame:
    df = pd.read_csv(TOPK_PATH)
    out = df[["k", "experiment", "best_R2_model", "best_R2", "best_R2_MAE"]].copy()
    out["Direction"] = out["experiment"].map({
        "hust_to_matr": "HUST -> MATR",
        "matr_to_hust": "MATR -> HUST",
    })
    out = out.rename(columns={
        "k": "k",
        "best_R2_model": "Best model",
        "best_R2": "Best R2",
        "best_R2_MAE": "MAE",
    })
    return out[["k", "Direction", "Best model", "Best R2", "MAE"]]


def make_coral_table() -> pd.DataFrame:
    df = pd.read_csv(CORAL_PATH)
    rows = []
    for experiment in ["hust_to_matr", "matr_to_hust"]:
        part = df[df["experiment"] == experiment].copy()
        none = part[part["adapter"] == "none"].iloc[0]
        cal = part[part["adapter"] == "residual_mean"].iloc[0]
        direction = "HUST -> MATR" if experiment == "hust_to_matr" else "MATR -> HUST"
        rows.append({
            "Direction": direction,
            "Setting": "CORAL only",
            "k target": 0,
            "R2": float(none["R2_mean"]),
            "MAE": float(none["MAE_mean"]),
        })
        rows.append({
            "Direction": direction,
            "Setting": "CORAL + target calibration",
            "k target": int(cal["k_target"]),
            "R2": float(cal["R2_mean"]),
            "MAE": float(cal["MAE_mean"]),
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
    topk = make_topk_table()
    coral = make_coral_table()

    topk_path = OUTPUT_DIR / "topk_cross_best_table.html"
    coral_path = OUTPUT_DIR / "coral_target_calibration_table.html"
    write_html(
        topk_path,
        title="Top-k Feature Sweep: Cross-Dataset Transfer",
        subtitle="Best cross-dataset result for each k. The table shows that top-k selection alone keeps R2 negative.",
        df=topk,
        r2_columns=["Best R2"],
        note="Color guide: green = closer to zero, yellow = mildly negative, red = strongly negative.",
    )
    write_html(
        coral_path,
        title="CORAL + Target-Side Calibration",
        subtitle="CORAL representation alignment before and after residual target calibration with k=20 labeled target cells.",
        df=coral,
        r2_columns=["R2"],
        note="The calibrated rows show the practical benefit of using a small labeled target set.",
    )

    summary_path = OUTPUT_DIR / "advisor_email_tables_summary.md"
    summary_path.write_text(
        "# Advisor Email Table Files\n\n"
        "Recommended attachments or screenshots:\n\n"
        f"1. `{topk_path.relative_to(PROJECT_ROOT)}`\n"
        f"2. `{coral_path.relative_to(PROJECT_ROOT)}`\n\n"
        "Suggested email sentence:\n\n"
        "> CSV dosyalarını daha okunabilir hale getirmek için iki renkli özet tablo oluşturdum: "
        "biri top-k cross-dataset sonuçlarını, diğeri CORAL + target calibration ile elde edilen iyileşmeyi gösteriyor.\n"
    )

    print(f"[save] {topk_path.relative_to(PROJECT_ROOT)}")
    print(f"[save] {coral_path.relative_to(PROJECT_ROOT)}")
    print(f"[save] {summary_path.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
