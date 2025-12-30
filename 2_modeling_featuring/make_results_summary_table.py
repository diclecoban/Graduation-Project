from pathlib import Path
import json
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_PATH = PROJECT_ROOT / "results_top8_metrics.json"
PLOTS_DIR = PROJECT_ROOT / "plots"
OUTPUT = PLOTS_DIR / "table_results_qdstd.png"

def load_results(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"Missing metrics JSON: {path}")
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)

def build_table_rows(data: dict) -> list[list[str]]:
    rows = []
    models = [
        ("Random Forest", "random_forest"),
        ("XGBoost", "xgboost"),
        ("CatBoost", "catboost"),
    ]
    cycles = ["25", "50", "100"]
    for label, key in models:
        for cycle in cycles:
            with_metrics = data[key]["with_qd_std"][cycle]
            without_metrics = data[key]["without_qd_std"][cycle]
            rows.append(
                [
                    label,
                    cycle,
                    f"{with_metrics['MAE']:.2f}",
                    f"{with_metrics['R2']:.2f}",
                    f"{without_metrics['MAE']:.2f}",
                    f"{without_metrics['R2']:.2f}",
                ]
            )
    return rows

def render_table(rows: list[list[str]]) -> None:
    PLOTS_DIR.mkdir(exist_ok=True)
    header = [
        "Model",
        "n_cycles",
        "MAE (Qd_std var)",
        "R2 (Qd_std var)",
        "MAE (Qd_std yok)",
        "R2 (Qd_std yok)",
    ]
    table_data = [header] + rows
    fig_height = max(2.5, 0.35 * len(rows) + 1.0)
    fig, ax = plt.subplots(figsize=(8, fig_height))
    ax.axis("off")
    table = ax.table(
        cellText=table_data,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.2)
    ax.set_title("Model vs n_cycles – Qd_std impact", pad=12)
    fig.tight_layout()
    fig.savefig(OUTPUT, dpi=200)
    plt.close(fig)
    print(f"Saved {OUTPUT}")

def main() -> None:
    data = load_results(RESULTS_PATH)
    rows = build_table_rows(data)
    render_table(rows)

if __name__ == "__main__":
    main()
