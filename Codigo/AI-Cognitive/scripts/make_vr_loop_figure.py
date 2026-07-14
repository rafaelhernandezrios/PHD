"""Diagrama de lazo cerrado: EEG -> indice de carga -> adaptacion del VR."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = "/sessions/gallant-elegant-hamilton/mnt/PHD/AI-Cognitive/paper/figures/fig_vr_loop.pdf"

fig, ax = plt.subplots(figsize=(7.2, 2.7))
ax.set_xlim(0, 12); ax.set_ylim(0, 5); ax.axis("off")

blue = "#2471a3"; green = "#1e8449"; gray = "#566573"; red = "#c0392b"


def box(x, y, w, h, text, color, fs=8):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                linewidth=1.4, edgecolor=color, facecolor=color + "20"))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color="black")


def arrow(x1, y1, x2, y2, color="black", lw=1.6):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=13,
                                 linewidth=lw, color=color, shrinkA=1, shrinkB=1))


# Top pipeline (perception): left -> right
box(0.2, 3.2, 2.2, 1.3, "Trainee at VR\nrobotic-surgery\nsimulator", blue, 8)
box(2.9, 3.4, 1.7, 1.0, "Wireless EEG\n(Fz, Pz, ...)", gray)
box(5.0, 3.4, 1.8, 1.0, "Features\n$\\theta/\\alpha$, band power", gray)
box(7.2, 3.4, 1.6, 1.0, "Per-subject\ncalibration", gray)
box(9.2, 3.2, 2.6, 1.3, "Classifier (RF)\nload level +\nCLI $\\theta$(Fz)/$\\alpha$(Pz)", green, 8)

arrow(2.4, 3.9, 2.9, 3.9)
arrow(4.6, 3.9, 5.0, 3.9)
arrow(6.8, 3.9, 7.2, 3.9)
arrow(8.8, 3.9, 9.2, 3.9)

# Bottom path (action / feedback): right -> left
box(8.5, 0.5, 3.3, 1.2, "Adaptation policy\nlow $\\rightarrow$ harder · optimal $\\rightarrow$ hold\nhigh $\\rightarrow$ easier / more guidance", red, 7.5)
box(3.4, 0.6, 3.6, 1.0, "VR update: scenario\ndifficulty, haptic guidance, pacing", red, 8)

arrow(10.5, 3.2, 10.5, 1.7, color=red)          # classifier -> policy (down)
arrow(8.5, 1.1, 7.0, 1.1, color=red)            # policy -> VR update (left)
arrow(3.4, 1.1, 1.3, 1.1, color=red)            # VR update -> left
arrow(1.3, 1.1, 1.3, 3.2, color=red)            # up into trainee box

ax.text(6.0, 4.55, "Perception: EEG $\\rightarrow$ cognitive-load estimate",
        ha="center", fontsize=8.5, color=green, style="italic")
ax.text(6.0, 0.05, "Action: close the loop by adapting the VR training in real time",
        ha="center", fontsize=8.5, color=red, style="italic")

plt.tight_layout()
plt.savefig(OUT, bbox_inches="tight")
print("saved", OUT)
