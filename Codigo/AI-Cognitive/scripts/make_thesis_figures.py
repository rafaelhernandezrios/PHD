"""Genera las figuras del Capitulo 5 de la tesis (etiquetas en espanol)."""
import os, json
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/sessions/gallant-elegant-hamilton/mnt/PHD/AI-Cognitive"
OUT = "/sessions/gallant-elegant-hamilton/mnt/Doctorado/Propuesta-de-Tesis-Rafa/Capitulo5/figuras"
os.makedirs(OUT, exist_ok=True)

df = pd.read_csv(f"{ROOT}/csv/eeg_window_features_aura.csv")
df = df[df.condition_4.isin(["normal", "low", "mid", "high"])].copy()
order = ["normal", "low", "mid", "high"]
xlab = ["reposo\n(baja)", "low\n(óptima)", "mid\n(óptima)", "high\n(alta)"]

# ---------- FIG CLI / bandas ----------
df["ft"] = df[["Fp1_rel_bp_theta", "Fp2_rel_bp_theta", "F3_rel_bp_theta",
               "Fz_rel_bp_theta", "F4_rel_bp_theta"]].mean(1)
df["pa"] = df[["P3_rel_bp_alpha", "Pz_rel_bp_alpha", "P4_rel_bp_alpha"]].mean(1)
df["cli"] = df["Fz_bp_theta"] / df["Pz_bp_alpha"].replace(0, np.nan)


def per_subj(col):
    rows = {c: [] for c in order}
    for s, g in df.groupby("subject_id"):
        base = g[g.condition_4 == "normal"][col].median()
        for c in order:
            v = g[g.condition_4 == c][col]
            if len(v):
                rows[c].append(v.median() - base)
    return (np.array([np.mean(rows[c]) for c in order]),
            np.array([np.std(rows[c]) / np.sqrt(len(rows[c])) for c in order]))


ftm, fts = per_subj("ft"); pam, pas = per_subj("pa"); clim, clis = per_subj("cli")
x = np.arange(4)
fig, ax = plt.subplots(1, 2, figsize=(7.6, 3.2))
ax[0].errorbar(x, ftm, yerr=fts, marker="o", color="#c0392b", lw=2, capsize=3,
               label="Theta frontal (rel.)")
ax[0].errorbar(x, pam, yerr=pas, marker="s", color="#2471a3", lw=2, capsize=3,
               label="Alpha parietal (rel.)")
ax[0].axhline(0, ls=":", c="gray", lw=1)
ax[0].set_xticks(x); ax[0].set_xticklabels(xlab, fontsize=7)
ax[0].set_ylabel("Δ potencia relativa vs reposo")
ax[0].set_title(r"(a) Firma de carga: $\theta\!\uparrow$, $\alpha\!\downarrow$", fontsize=9)
ax[0].legend(fontsize=7, loc="center left")
ax[1].errorbar(x, clim, yerr=clis, marker="D", color="#6c3483", lw=2, capsize=3)
ax[1].axhline(0, ls=":", c="gray", lw=1)
ax[1].set_xticks(x); ax[1].set_xticklabels(xlab, fontsize=7)
ax[1].set_ylabel(r"Δ CLI = $\theta$(Fz)/$\alpha$(Pz)")
ax[1].set_title("(b) El índice de carga sube en tarea", fontsize=9)
plt.tight_layout(); plt.savefig(f"{OUT}/fig_cli_bandas.pdf", bbox_inches="tight"); plt.close()

# ---------- FIG montaje ----------
d = json.load(open(f"{ROOT}/csv/ablation_results.json"))
monts = ["all8", "fp6", "fzpz2"]
labels = ["8 can.\n(218)", "6 can. F+P\n(166)", "2 can. Fz+Pz\n(52)"]
binacc = [d["binary_" + m]["acc"] for m in monts]
triacc = [d["hier_" + m]["acc"] for m in monts]
fig, ax = plt.subplots(1, 2, figsize=(7.6, 3.2)); xx = np.arange(3); w = 0.36
ax[0].bar(xx - w / 2, binacc, w, label="Binaria (reposo/tarea)", color="#6fa8dc")
ax[0].bar(xx + w / 2, triacc, w, label="3 clases (baja/ópt/alta)", color="#93c47d")
ax[0].axhline(0.388, ls=":", c="gray", lw=1)
ax[0].text(1.35, 0.40, "azar 3 cl.", fontsize=6, color="gray")
ax[0].set_xticks(xx); ax[0].set_xticklabels(labels, fontsize=7)
ax[0].set_ylabel("Exactitud LOSO"); ax[0].set_ylim(0, 1)
ax[0].set_title("(a) Ablación de montaje (sin FAA)", fontsize=9)
ax[0].legend(fontsize=6, loc="upper right")
for i, (b, t) in enumerate(zip(binacc, triacc)):
    ax[0].text(i - w / 2, b + .02, f"{b:.2f}", ha="center", fontsize=6)
    ax[0].text(i + w / 2, t + .02, f"{t:.2f}", ha="center", fontsize=6)
for s, g in df.groupby("subject_id"):
    base = g[g.condition_4 == "normal"]["cli"]; mu, sd = base.mean(), base.std()
    sd = sd if sd > 1e-9 else 1
    df.loc[g.index, "cliz"] = (g["cli"] - mu) / sd
dd = [np.clip(df[df.condition_4 == c]["cliz"].dropna(), -4, 8) for c in order]
bp = ax[1].boxplot(dd, showfliers=False, patch_artist=True, widths=0.6)
for p, c in zip(bp["boxes"], ["#cccccc", "#9fc5e8", "#6fa8dc", "#e06666"]):
    p.set_facecolor(c)
ax[1].set_xticklabels(["baja", "ópt(low)", "ópt(mid)", "alta"], fontsize=7, rotation=15)
ax[1].set_ylabel(r"CLI z-score  $\theta$(Fz)/$\alpha$(Pz)")
ax[1].axhline(0, ls=":", c="gray", lw=1)
ax[1].set_title(r"(b) Índice CLI por nivel ($\rho$=0.32)", fontsize=9)
plt.tight_layout(); plt.savefig(f"{OUT}/fig_montaje.pdf", bbox_inches="tight"); plt.close()

# ---------- FIG tres clases ----------
t = json.load(open(f"{ROOT}/csv/three_class_results.json"))
names = ["baja", "óptima", "alta"]
fig, ax = plt.subplots(1, 2, figsize=(7.6, 3.1))
cm = np.array(t["hier"]["cm"], float); cmn = cm / cm.sum(1, keepdims=True)
ax[0].imshow(cmn, cmap="Blues", vmin=0, vmax=1)
ax[0].set_xticks(range(3)); ax[0].set_xticklabels(names)
ax[0].set_yticks(range(3)); ax[0].set_yticklabels(names)
ax[0].set_xlabel("predicho"); ax[0].set_ylabel("verdadero")
ax[0].set_title("(a) Confusión RF jerárquico", fontsize=9)
for i in range(3):
    for j in range(3):
        ax[0].text(j, i, f"{cmn[i, j]:.2f}", ha="center", va="center",
                   color="white" if cmn[i, j] > 0.5 else "black", fontsize=9)
ordk = ["ordinal", "rf", "hgb", "hier"]; lab = ["Ordinal", "RF bal.", "GB bal.", "Jerárq."]
cols = ["#bbbbbb", "#6fa8dc", "#f6b26b", "#93c47d"]; xr = np.arange(3); ww = 0.2
for i, k in enumerate(ordk):
    rc = t[k]["recall"]; vals = [rc["baja"], rc["optima"], rc["alta"]]
    ax[1].bar(xr + (i - 1.5) * ww, vals, ww, label=lab[i], color=cols[i])
ax[1].set_xticks(xr); ax[1].set_xticklabels(names)
ax[1].set_ylabel("recall por clase"); ax[1].set_ylim(0, 1)
ax[1].set_title("(b) Recall por clase y método", fontsize=9)
ax[1].legend(fontsize=6, ncol=2, loc="upper right")
plt.tight_layout(); plt.savefig(f"{OUT}/fig_tresclases.pdf", bbox_inches="tight"); plt.close()

print("OK:", sorted(os.listdir(OUT)))
