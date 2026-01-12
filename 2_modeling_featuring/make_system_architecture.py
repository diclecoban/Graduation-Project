"""
Create a presentation-friendly system architecture diagram using Matplotlib.

Usage:
    python 2_modeling_featuring/make_system_architecture.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT = PROJECT_ROOT / "plots" / "system_architecture.png"

BOXES = [
    ("Data sources", "Stanford MAT\nBatch 1 & 2"),
    ("Data prep", "build_batch_from_mat.py"),
    ("Feature engineering", "ExtractDQdVFeatures +\nbuild_features_top8"),
    ("Splitting", "split_train_val_test.py"),
    ("Modeling", "RF / XGB / CatBoost /\nElasticNet + CV + Conformal"),
    ("Reporting", "Tables, plots,\nhold-out vs CV"),
]


def main() -> None:
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis("off")

    width = 0.12
    height = 0.28
    y = 0.5
    x_positions = [0.05, 0.21, 0.37, 0.53, 0.69, 0.85]
    colors = ["#1f2d4f", "#e9eef7", "#e9eef7", "#e9eef7", "#e9eef7", "#1f2d4f"]
    text_colors = ["white"] + ["#111d34"] * 4 + ["white"]

    for (title, desc), x, color, txt_color in zip(BOXES, x_positions, colors, text_colors):
        box = FancyBboxPatch(
            (x, y - height / 2),
            width,
            height,
            boxstyle="round,pad=0.015",
            linewidth=1.2,
            edgecolor="#d7d9e1",
            facecolor=color,
        )
        ax.add_patch(box)
        ax.text(
            x + width / 2,
            y + 0.06,
            title,
            ha="center",
            va="center",
            fontsize=10,
            color=txt_color,
            fontweight="bold",
        )
        ax.text(
            x + width / 2,
            y - 0.05,
            desc,
            ha="center",
            va="center",
            fontsize=8.5,
            color=txt_color,
        )

    for i in range(len(BOXES) - 1):
        start_x = x_positions[i] + width
        end_x = x_positions[i + 1]
        ax.annotate(
            "",
            xy=(end_x, y),
            xytext=(start_x, y),
            arrowprops=dict(arrowstyle="->", color="#111d34", lw=1.5),
        )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("System architecture overview", fontsize=14, color="#111d34")
    fig.tight_layout(rect=[0.02, 0.05, 0.98, 0.95])

    OUTPUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUTPUT, dpi=250)
    plt.close(fig)
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()
