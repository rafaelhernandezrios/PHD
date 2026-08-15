"""Figuras del Capitulo 5 de la tesis (modulo de IA cognitiva), en espanol.

Reescrito en agosto de 2026. La version anterior tenia dos defectos que
invalidaban dos de sus tres figuras:

  - rutas absolutas de otra maquina;
  - asumia que el dataset de aritmetica/Stroop tenia electrodos parietales
    (P3, Pz, P4). No los tiene: es un OpenBCI Cyton de 8 canales sobre
    Fp1 Fp2 F7 F3 Fz F4 F8 C4. Lo que aquella figura llamaba Pz era F8, de
    modo que el "CLI" que dibujaba no era el cociente frontal-parietal.

Por eso la ablacion de montaje y el indice CLI se calculan ahora sobre
eegmat, que si trae el 10-20 completo. Salida:

  Capitulo5/figuras/fig_descomposicion.pdf
  Capitulo5/figuras/fig_montaje.pdf
  Capitulo5/figuras/fig_tresclases.pdf
  Capitulo5/figuras/fig_ordinal.pdf
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
TESIS = Path("/Users/rafael/Documents/Proyectos Personales/Doctorado/Tesis/"
             "Propuesta-de-Tesis-Rafa")
OUT = TESIS / "Capitulo5" / "figuras"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif", "font.size": 9, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7.5,
    "savefig.dpi": 300, "savefig.bbox": "tight",
})

ROJO, AZUL, MORADO, VERDE, GRIS = "#c0392b", "#2471a3", "#6c3483", "#4f8a3d", "#8a8a8a"


# ---------------------------------------------------------------- figura 1
def fig_descomposicion():
    """Que mitad del cociente theta/alpha hace el trabajo."""
    d = json.loads((ROOT / "csv" / "cli_decomposition.json").read_text())
    ez = d["eegmat"]
    rng = np.random.default_rng(0)

    fig, ax = plt.subplots(1, 2, figsize=(7.4, 3.1),
                           gridspec_kw={"width_ratios": [1.0, 1.25]})

    # (a) eegmat: CLI y sus dos mitades, un punto por sujeto
    partes = [("CLI", r"CLI  $\theta_{Fz}/\alpha_{Pz}$", MORADO),
              ("theta_Fz", r"$\theta$(Fz)", ROJO),
              ("alpha_Pz", r"$\alpha$(Pz)", AZUL)]
    import matplotlib.transforms as mtransforms
    tr = mtransforms.blended_transform_factory(ax[0].transData, ax[0].transAxes)
    for i, (k, lab, col) in enumerate(partes):
        r = np.array(ez[k]["ratios"])
        ax[0].scatter(rng.normal(i, 0.07, len(r)), r, s=16, color=col,
                      alpha=0.7, edgecolor="black", linewidth=0.3, zorder=3)
        med = np.median(r)
        ax[0].hlines(med, i - 0.28, i + 0.28, color="black", lw=1.8, zorder=4)
        ax[0].text(i, 0.95, f"{med:.2f}$\\times$", ha="center", fontsize=8.5,
                   transform=tr)
        ax[0].text(i, 0.89, f"p={ez[k]['wilcoxon_p']:.1g}", ha="center",
                   fontsize=7, color="#555555", transform=tr)
    ax[0].axhline(1.0, ls=":", c=GRIS, lw=1)
    ax[0].set_yscale("log")
    ax[0].set_ylim(0.1, 22)
    ax[0].set_yticks([0.25, 0.5, 1, 2, 4])
    ax[0].set_yticklabels(["0.25", "0.5", "1", "2", "4"])
    ax[0].set_xticks(range(3))
    ax[0].set_xticklabels([p[1] for p in partes], fontsize=8)
    ax[0].set_ylabel("tarea / reposo (por sujeto)")
    ax[0].set_title("(a) eegmat, 36 sujetos", fontsize=9.5)

    # (b) AS: theta y alpha canal a canal
    az = d["arithmetic_stroop"]
    chans = az["_montage"]
    th = [az["theta"][c]["median_task_over_rest"] for c in chans]
    al = [az["alpha"][c]["median_task_over_rest"] for c in chans]
    x = np.arange(len(chans))
    w = 0.38
    ax[1].bar(x - w / 2, th, w, color=ROJO, edgecolor="black", linewidth=0.4,
              label=r"$\theta$ (tarea/reposo)")
    ax[1].bar(x + w / 2, al, w, color=AZUL, edgecolor="black", linewidth=0.4,
              label=r"$\alpha$ (tarea/reposo)")
    ax[1].axhline(1.0, ls=":", c=GRIS, lw=1)
    ax[1].set_xticks(x)
    ax[1].set_xticklabels(chans, fontsize=7.5)
    ax[1].set_ylim(0, 1.75)
    ax[1].set_ylabel("cociente respecto al reposo")
    ax[1].set_title("(b) Aritmetica/Stroop, 15 sujetos (montaje frontal)",
                    fontsize=9.5)
    ax[1].legend(loc="upper right", framealpha=0.9)
    # marcar los dos frontopolares, contaminados por parpadeo
    ax[1].annotate("frontopolares\n(artefacto ocular)", xy=(0.35, 1.44),
                   xytext=(2.6, 1.55), fontsize=6.5, color="#555555",
                   ha="center", va="center",
                   arrowprops=dict(arrowstyle="->", color="#555555", lw=0.7))

    fig.tight_layout()
    fig.savefig(OUT / "fig_descomposicion.pdf")
    plt.close(fig)


# ---------------------------------------------------------------- figura 2
def fig_montaje():
    """Ablacion de montaje sobre eegmat, mas el ascenso del CLI por sujeto."""
    d = json.loads((ROOT / "csv" / "zyma_montage_cli.json").read_text())
    dec = json.loads((ROOT / "csv" / "cli_decomposition.json").read_text())

    nombres = {
        "19 ch (full 10-20)": "19 canales\n(10–20 completo)",
        "8 ch (frontal + parietal)": "8 canales\n(diadema)",
        "6 ch (Fz,F3,F4,Pz,P3,P4)": "6 canales\nfrontal–parietal",
        "2 ch (Fz,Pz)": "2 canales\n(Fz, Pz)",
    }
    keys = list(nombres)
    acc = [d[k]["acc"] for k in keys]
    std = [d[k]["acc_std"] for k in keys]
    nfe = [d[k]["n_features"] for k in keys]

    fig, ax = plt.subplots(1, 2, figsize=(7.4, 3.0),
                           gridspec_kw={"width_ratios": [1.5, 1.0]})

    y = np.arange(len(keys))
    ax[0].barh(y, acc, xerr=std, color="#3b6fb0", edgecolor="black",
               linewidth=0.4, capsize=2.5, height=0.6)
    for i, a in enumerate(acc):
        ax[0].text(a + std[i] + 0.012, i, f"{a:.3f}", va="center", fontsize=8)
    ax[0].set_yticks(y)
    ax[0].set_yticklabels([f"{nombres[k]}\n({n} carac.)" for k, n in zip(keys, nfe)],
                          fontsize=7.5)
    ax[0].invert_yaxis()
    ax[0].set_xlim(0.55, 0.99)
    ax[0].set_xlabel("exactitud binaria (LOSO + calibracion)")
    ax[0].set_title("(a) Ablacion de montaje, eegmat", fontsize=9.5)

    r = np.array(dec["eegmat"]["CLI"]["ratios"])
    rng = np.random.default_rng(0)
    ax[1].axhline(1.0, ls=":", c=GRIS, lw=1)
    ax[1].scatter(rng.normal(0, 0.045, len(r)), r, s=20, color=ROJO,
                  alpha=0.75, edgecolor="black", linewidth=0.35)
    ax[1].hlines(np.median(r), -0.13, 0.13, color="black", lw=1.8)
    ax[1].set_xlim(-0.3, 0.3)
    ax[1].set_xticks([])
    ax[1].set_ylabel("CLI en tarea / CLI en reposo")
    ax[1].set_title(r"(b) $\theta$(Fz)/$\alpha$(Pz) sube 1.93x en 31/36",
                    fontsize=9.5)

    fig.tight_layout()
    fig.savefig(OUT / "fig_montaje.pdf")
    plt.close(fig)


# ---------------------------------------------------------------- figura 3
def fig_tresclases():
    t = json.loads((ROOT / "csv" / "three_class_results.json").read_text())
    clases = ["baja", "optima", "alta"]

    fig, ax = plt.subplots(1, 2, figsize=(7.4, 3.0))

    cm = np.array(t["hier"]["cm"], float)
    cmn = cm / cm.sum(1, keepdims=True)
    ax[0].imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax[0].set_xticks(range(3)); ax[0].set_xticklabels(clases)
    ax[0].set_yticks(range(3)); ax[0].set_yticklabels(clases)
    ax[0].set_xlabel("predicho"); ax[0].set_ylabel("verdadero")
    ax[0].set_title("(a) Confusion, RF jerarquico", fontsize=9.5)
    for i in range(3):
        for j in range(3):
            ax[0].text(j, i, f"{cmn[i, j]:.2f}", ha="center", va="center",
                       color="white" if cmn[i, j] > 0.5 else "black", fontsize=9)

    orden = ["ordinal", "rf", "hgb", "hier"]
    etiq = ["Ordinal", "RF bal.", "GB bal.", "Jerarquico"]
    cols = ["#bbbbbb", "#6fa8dc", "#f6b26b", VERDE]
    x = np.arange(3); w = 0.2
    for i, k in enumerate(orden):
        rc = t[k]["recall"]
        ax[1].bar(x + (i - 1.5) * w, [rc["baja"], rc["optima"], rc["alta"]], w,
                  label=etiq[i], color=cols[i], edgecolor="black", linewidth=0.3)
    ax[1].set_xticks(x); ax[1].set_xticklabels(clases)
    ax[1].set_ylabel("recall por clase"); ax[1].set_ylim(0, 1)
    ax[1].set_title("(b) Recall por clase y metodo", fontsize=9.5)
    ax[1].legend(ncol=2, loc="upper right", framealpha=0.9)

    fig.tight_layout()
    fig.savefig(OUT / "fig_tresclases.pdf")
    plt.close(fig)


# ---------------------------------------------------------------- figura 4
def fig_ordinal():
    df = pd.read_csv(ROOT / "csv" / "ordinal_predictions.csv")
    niveles = [0, 1, 2, 3]
    etiq = ["reposo", "bajo", "medio", "alto"]
    rng = np.random.default_rng(0)

    fig, ax = plt.subplots(1, 2, figsize=(7.4, 3.0))

    for n in niveles:
        v = df.loc[df.y_true == n, "y_pred"]
        ax[0].scatter(rng.normal(n, 0.08, len(v)), v, s=3, alpha=0.06,
                      color=AZUL, edgecolor="none")
    mu = [df.loc[df.y_true == n, "y_pred"].mean() for n in niveles]
    sd = [df.loc[df.y_true == n, "y_pred"].std() for n in niveles]
    ax[0].errorbar(niveles, mu, yerr=sd, fmt="o", color=ROJO, lw=1.6,
                   capsize=3, zorder=5, ms=5)
    ax[0].plot([-0.3, 3.3], [-0.3, 3.3], ls="--", c=GRIS, lw=1)
    ax[0].set_xticks(niveles); ax[0].set_xticklabels(etiq)
    ax[0].set_xlabel("nivel real"); ax[0].set_ylabel("puntuacion predicha")
    ax[0].set_ylim(-0.4, 3.4)
    ax[0].set_title("(a) Regresion ordinal, LOSO", fontsize=9.5)

    # dispersion por sujeto del rho de Spearman
    rhos = []
    for s, g in df.groupby("subject_id"):
        rhos.append(g[["y_true", "y_pred"]].corr(method="spearman").iloc[0, 1])
    rhos = np.array(rhos)
    ax[1].bar(np.arange(len(rhos)), np.sort(rhos)[::-1], color="#6fa8dc",
              edgecolor="black", linewidth=0.4)
    ax[1].axhline(np.mean(rhos), ls="--", c=ROJO, lw=1.2,
                  label=f"media {np.mean(rhos):.2f}")
    ax[1].axhline(0, c="black", lw=0.8)
    ax[1].set_xlabel("sujetos (ordenados)")
    ax[1].set_ylabel(r"Spearman $\rho$ intra-sujeto")
    ax[1].set_xticks([])
    ax[1].set_title("(b) Consistencia por sujeto", fontsize=9.5)
    ax[1].legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(OUT / "fig_ordinal.pdf")
    plt.close(fig)
    return float(np.mean(rhos)), float(np.std(rhos))


if __name__ == "__main__":
    fig_descomposicion()
    fig_montaje()
    fig_tresclases()
    m, s = fig_ordinal()
    print(f"rho por sujeto: {m:.3f} +- {s:.3f}")
    print("OK:", sorted(p.name for p in OUT.glob("*.pdf")))
