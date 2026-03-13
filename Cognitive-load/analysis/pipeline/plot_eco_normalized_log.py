"""
Dot plot (scatter) of normalized theta/alpha ratio per session for the ecological paradigm.
IEEE single-column compatible, log10 Y-scale, baseline at y=1.
Uses only matplotlib (no seaborn).
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Data
sessions = ["S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8"]
ratios = [7.85, 1.87, 2.51, 3.14, 4.08, 2.14, 0.71, 0.33]

# Output
BASE_DIR = Path(__file__).resolve().parents[2]
OUTPUT_PATH = BASE_DIR / "output" / "analysis_output" / "eco_normalized_ratio_log.png"

# IEEE single-column width (approx 3.5 in), height to taste
FIG_W = 3.5
FIG_H = 2.5

# Font sizes (8-9 pt style)
FONT_SIZE = 8
TICK_SIZE = 8
ANNOT_SIZE = 7

plt.rcParams.update({"font.size": FONT_SIZE})

fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))

x = np.arange(len(sessions))
y = np.array(ratios, dtype=float)

# Scatter
ax.scatter(x, y, color="#1976d2", s=28, zorder=3, edgecolors="#333", linewidths=0.8)

# Optional: annotate value above each point (2 decimals)
for i, (xi, yi) in enumerate(zip(x, y)):
    ax.annotate(
        f"{yi:.2f}",
        (xi, yi),
        xytext=(0, 6),
        textcoords="offset points",
        ha="center",
        fontsize=ANNOT_SIZE,
        color="black",
    )

# Baseline at y = 1
ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=1, alpha=0.8, zorder=1)

# Log scale Y
ax.set_yscale("log")
ax.set_xlim(-0.5, len(sessions) - 0.5)
# Avoid log(0); set ylim if needed (0.33 is min, so 0.1–10 or similar is fine)
ax.set_ylim(0.2, 15)

ax.set_xticks(x)
ax.set_xticklabels(sessions, rotation=45, ha="right")
ax.set_xlabel("Session", fontsize=FONT_SIZE)
ax.set_ylabel(
    "Normalized Theta/Alpha (F region / P region)\n(task / baseline)",
    fontsize=FONT_SIZE,
)
ax.tick_params(axis="both", labelsize=TICK_SIZE)

# Light grid on Y only
ax.yaxis.grid(True, linestyle="-", alpha=0.25)
ax.set_axisbelow(True)

ax.set_facecolor("white")
fig.patch.set_facecolor("white")
for spine in ax.spines.values():
    spine.set_color("black")

plt.tight_layout()
fig.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight", facecolor="white")
plt.close()

print(f"Figure saved: {OUTPUT_PATH}")
