"""Create a single table comparing hold-out vs cross-validation metrics."""

from pathlib import Path
import json
import math

import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parent.parent
HOLDOUT_PATH = PROJECT_ROOT / "results_top8_metrics.json"
CV_PATH = PROJECT_ROOT / "results_top8_cv_metrics.json"
PLOTS_DIR = PROJECT_ROOT / "plots"
OUTPUT = PLOTS_DIR / "table_holdout_vs_cv.png"

MODELS = [
    ("Random Forest", "random_forest"),
    ("XGBoost", "xgboost"),
    ("CatBoost", "catboost"),
    ("ElasticNet", "elastic_net"),
]
CYCLES = ("25", "50", "100")
FEATURE_KEY = "with_qd_std"


def load_json(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"Missing JSON file: {path}")
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def format_cv(value_dict: dict | None, metric: str) -> str:
    if not value_dict or metric not in value_dict:
        return "-"
    mean = value_dict[metric].get("mean")
    std = value_dict[metric].get("std")
    if mean is None:
        return "-"
    if std is None or math.isclose(std, 0.0):
        return f"{mean:.2f}"
    return f"{mean:.2f} ± {std:.2f}"


def build_rows(holdout: dict, cv: dict) -> list[list[str]]:
    rows = []
    for model_label, key in MODELS:
        holdout_model = holdout.get(key, {}).get(FEATURE_KEY, {})
        cv_model = cv.get(key, {}).get(FEATURE_KEY, {})
        for cycle in CYCLES:
            holdout_metrics = holdout_model.get(cycle)
            cv_metrics = cv_model.get(int(cycle)) or cv_model.get(cycle)
            if not holdout_metrics and not cv_metrics:
                continue
            rows.append(
                [
                    model_label,
                    cycle,
                    f"{holdout_metrics['MAE']:.2f}" if holdout_metrics else "-",
                    format_cv(cv_metrics, "MAE"),
                    f"{holdout_metrics['R2']:.2f}" if holdout_metrics else "-",
                    format_cv(cv_metrics, "R2"),
                    f"{holdout_metrics['MAPE']:.2f}" if holdout_metrics else "-",
                    format_cv(cv_metrics, "MAPE"),
                    f"{holdout_metrics['SMAPE']:.2f}" if holdout_metrics else "-",
                    format_cv(cv_metrics, "SMAPE"),
                ]
            )
    return rows


def render_table(rows: list[list[str]]) -> None:
    if not rows:
        raise SystemExit("No rows to render for hold-out vs CV table.")

    PLOTS_DIR.mkdir(exist_ok=True)
    header = [
        "Model",
        "n_cycles",
        "Hold-out MAE",
        "CV MAE (mean ± std)",
        "Hold-out R2",
        "CV R2 (mean ± std)",
        "Hold-out MAPE",
        "CV MAPE (mean ± std)",
        "Hold-out SMAPE",
        "CV SMAPE (mean ± std)",
    ]
    table_data = [header] + rows
    fig_height = max(2.5, 0.35 * len(rows) + 1.0)
    fig, ax = plt.subplots(figsize=(14, fig_height))
    ax.axis("off")
    table = ax.table(cellText=table_data, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.1)
    ax.set_title("Hold-out vs Grouped CV (Qd_std retained)", pad=12)
    fig.tight_layout()
    fig.savefig(OUTPUT, dpi=200)
    plt.close(fig)
    print(f"Saved {OUTPUT}")


def main() -> None:
    holdout = load_json(HOLDOUT_PATH)
    cv = load_json(CV_PATH)
    rows = build_rows(holdout, cv)
    render_table(rows)


if __name__ == "__main__":
    main()
