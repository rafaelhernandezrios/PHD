"""Closed-loop diagram: EEG -> load score -> difficulty -> trainer.

Drawn one column wide (IEEEtran) so it does not cost a full-width float, and
kept in step with the paper: the score comes from the ordinal regressor and not
from the random forest, the guards sit between score and action, and the
performance gate closes a second path from the trainer's own error telemetry.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = Path(__file__).resolve().parent.parent / "paper" / "figures" / "fig_vr_loop.pdf"

plt.rcParams.update({"font.family": "serif", "savefig.bbox": "tight",
                     "savefig.dpi": 300})

BLUE, GREEN, GRAY, RED = "#2471a3", "#1e8449", "#566573", "#c0392b"

fig, ax = plt.subplots(figsize=(3.45, 3.0))   # one IEEEtran column
ax.set_xlim(0, 10); ax.set_ylim(0, 9.4); ax.axis("off")


def box(x, y, w, h, text, color, fs=6.4):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.02,rounding_size=0.14",
                                linewidth=1.1, edgecolor=color,
                                facecolor=color + "1e"))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, color="black", linespacing=1.35)


def arrow(x1, y1, x2, y2, color="black", lw=1.2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=9, linewidth=lw, color=color,
                                 shrinkA=1, shrinkB=1))


# ---- perception: published front end + companion classifier -----------------
ax.text(5.0, 9.15, "perception", ha="center", fontsize=6.8, style="italic",
        color=GREEN)
box(0.7, 7.45, 4.0, 1.2, "wireless EEG, 8 ch\n4 s window / 2 s stride", GRAY)
box(5.3, 7.45, 4.0, 1.2, "band powers, $\\theta/\\alpha$\nsubject calibration", GRAY)
box(2.3, 5.85, 5.4, 1.15, "ordinal regressor\nload score $s_t\\in[0,3]$", GREEN)
arrow(4.7, 8.05, 5.3, 8.05)
arrow(7.3, 7.45, 6.2, 7.00)

# ---- action: this paper -----------------------------------------------------
ax.text(7.9, 5.42, "action  (this paper)", ha="center", fontsize=6.8,
        style="italic", color=RED)
box(1.1, 3.85, 7.8, 1.25,
    "smoothing, dead-band,\npersistence, refractory", RED)
box(2.9, 2.30, 4.2, 1.05, "difficulty $d\\in[0,1]$", RED)
box(0.7, 0.40, 8.6, 1.30,
    "trainer: motion scaling, rendered\nforce, tolerances, time budget", BLUE)
arrow(5.0, 5.85, 5.0, 5.10, RED)
arrow(5.0, 3.85, 5.0, 3.35, RED)
arrow(5.0, 2.30, 5.0, 1.70, RED)

# ---- performance gate: second path, telemetry -> guards ---------------------
ax.plot([9.6, 9.6], [1.05, 4.48], color=GRAY, lw=1.0, zorder=0)
arrow(9.6, 1.05, 9.6, 4.48, GRAY, lw=1.0)
arrow(9.6, 4.48, 8.9, 4.48, GRAY, lw=1.0)
ax.text(9.25, 2.75, "error telemetry", rotation=90, ha="center", va="center",
        fontsize=5.8, color=GRAY)

# ---- the trainee closes the loop -------------------------------------------
ax.plot([0.25, 0.25], [1.05, 8.05], color=BLUE, lw=1.0, zorder=0)
arrow(0.25, 1.05, 0.25, 8.05, BLUE, lw=1.0)
arrow(0.25, 8.05, 0.7, 8.05, BLUE, lw=1.0)
ax.text(0.58, 4.7, "trainee", rotation=90, ha="center", va="center",
        fontsize=5.8, color=BLUE)

OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT)
print(f"[fig] {OUT}")
