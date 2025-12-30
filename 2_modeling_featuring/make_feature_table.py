from pathlib import Path

import matplotlib.pyplot as plt

plt.rcParams["font.size"] = 9

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
cycles = ["25 cycles", "50 cycles", "100 cycles"]
data = [["X" for _ in cycles] for _ in FEATURES]

fig, ax = plt.subplots(figsize=(4.2, len(FEATURES) * 0.35 + 1.2))
ax.axis("off")
table_data = [["Feature"] + cycles] + [[feat] + row for feat, row in zip(FEATURES, data)]
table = ax.table(cellText=table_data, loc="center", cellLoc="center")
table.auto_set_font_size(False)
table.set_fontsize(8)
table.scale(1, 1.2)
ax.set_title("Eight-Feature Setup vs n_cycles", pad=12)
fig.tight_layout()
plots_dir = Path(__file__).resolve().parent.parent / "plots"
plots_dir.mkdir(exist_ok=True)
fig.savefig(plots_dir / "table_features_cycles.png", dpi=200)
plt.close(fig)
