"""Figuras del Capitulo 4 de la tesis (consola adaptativa), en espanol.

Reutiliza la politica y el escenario de adaptive_replay.py para no mantener dos
implementaciones del lazo. Salida:

  Capitulo4/figuras/fig_lazo_traza.pdf      traza de sesion + ablacion de guardas
  Capitulo4/figuras/fig_lazo_asimetria.pdf  asimetria del lazo + barrido de umbral
"""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from adaptive_replay import (  # noqa: E402
    ALTA, BAJA, HIGH, LOW, MID, OPTIMA, REST, STRIDE_SEC, TRUE_ZONE,
    PolicyConfig, build_ramp, run_all, run_subject, summarise,
)

ROOT = HERE.parent
TESIS = Path("/Users/rafael/Documents/Proyectos Personales/Doctorado/Tesis/"
             "Propuesta-de-Tesis-Rafa")
OUT = TESIS / "Capitulo4" / "figuras"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif", "font.size": 9, "axes.labelsize": 9,
    "axes.titlesize": 9.5, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 7.5, "savefig.dpi": 300, "savefig.bbox": "tight",
})

ROJO, AZUL, GRIS = "#c0392b", "#2471a3", "#8a8a8a"
COND_ES = {REST: "reposo", LOW: "bajo", MID: "medio", HIGH: "alto"}
COND_COLOR = {REST: "#eeeeee", LOW: "#dbe9f6", MID: "#c3dcf0", HIGH: "#f5d6d2"}

# Guardas anti-oscilacion, acumulativas. Misma secuencia que la ablacion del
# paper: sin nada, +suavizado, +banda muerta, +persistencia y refractario.
GUARDAS = [
    ("sin guardas", dict(ema_lambda=0.0, hysteresis=0.0, persist_windows=1,
                         refractory_sec=0.0)),
    ("+ suavizado", dict(hysteresis=0.0, persist_windows=1, refractory_sec=0.0)),
    ("+ banda muerta", dict(persist_windows=1, refractory_sec=0.0)),
    ("+ persist. y refr.", dict()),
]


def cargar():
    return pd.read_csv(ROOT / "csv" / "ordinal_predictions.csv")


def ablacion(df, base):
    filas = []
    for nombre, over in GUARDAS:
        res = summarise(run_all(df, replace(base, **over)))
        filas.append((nombre, res["switches_per_min"], res["actions_per_min"],
                      res["zone_agreement"]))
    return filas


# ------------------------------------------------------------------ figura 1
def fig_traza(df, base, filas):
    # sujeto con la concordancia de zona mediana: el caso representativo
    per = run_all(df, base)
    orden = sorted(per, key=lambda s: per[s]["zone_agreement"])
    sid = orden[len(orden) // 2]
    r = per[sid]
    tr, truth = r["_trace"], r["_truth"]
    t = np.array([x[0] for x in tr]) / 60.0
    ema = np.array([x[2] for x in tr])
    d = np.array([x[4] for x in tr])

    fig, ax = plt.subplots(1, 2, figsize=(7.4, 3.0),
                           gridspec_kw={"width_ratios": [1.55, 1.0]})

    # (a) traza de la sesion
    ini = 0
    for i in range(1, len(truth) + 1):
        if i == len(truth) or truth[i] != truth[ini]:
            ax[0].axvspan(t[ini], t[i - 1], color=COND_COLOR[truth[ini]], lw=0)
            ax[0].text((t[ini] + t[i - 1]) / 2, 2.92, COND_ES[truth[ini]],
                       ha="center", fontsize=6.5, color="#444444")
            ini = i
    ax[0].plot(t, ema, color=AZUL, lw=1.2, label="carga estimada (suavizada)")
    ax[0].axhline(base.theta_lo, ls=":", c=GRIS, lw=0.9)
    ax[0].axhline(base.theta_hi, ls=":", c=GRIS, lw=0.9)
    ax[0].set_ylim(0, 3.1)
    ax[0].set_xlabel("tiempo de sesion (min)")
    ax[0].set_ylabel("puntuacion de carga (0–3)")
    ax2 = ax[0].twinx()
    ax2.plot(t, d, color=ROJO, lw=1.4, label="dificultad $d$")
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel("dificultad $d$", color=ROJO)
    ax2.tick_params(axis="y", colors=ROJO)
    h1, l1 = ax[0].get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax[0].legend(h1 + h2, l1 + l2, loc="lower left", framealpha=0.9)
    ax[0].set_title(f"(a) Lazo cerrado, sujeto de concordancia mediana "
                    f"({r['zone_agreement']:.2f})")

    # (b) ablacion de guardas
    nombres = [f[0] for f in filas]
    sw = [f[1]["mean"] for f in filas]
    ac = [f[2]["mean"] for f in filas]
    sw_e = [f[1]["std"] for f in filas]
    ac_e = [f[2]["std"] for f in filas]
    x = np.arange(len(filas)); w = 0.38
    ax[1].bar(x - w / 2, sw, w, yerr=sw_e, capsize=2.5, color="#6fa8dc",
              edgecolor="black", linewidth=0.4, label="cambios de zona")
    ax[1].bar(x + w / 2, ac, w, yerr=ac_e, capsize=2.5, color=ROJO,
              edgecolor="black", linewidth=0.4, label="acciones de dificultad")
    ax[1].set_yscale("log")
    ax[1].set_xticks(x)
    ax[1].set_xticklabels(nombres, rotation=20, ha="right", fontsize=7)
    ax[1].set_ylabel("eventos por minuto")
    ax[1].legend(loc="upper right", framealpha=0.9)
    ax[1].set_title("(b) Guardas anti-oscilacion")

    fig.tight_layout()
    fig.savefig(OUT / "fig_lazo_traza.pdf")
    plt.close(fig)
    return sid, per


# ------------------------------------------------------------------ figura 2
def fig_asimetria(df, base, per):
    # (a) precision y recall por direccion, un punto por sujeto
    def vals(key):
        return np.array([v[key] for v in per.values() if v[key] is not None])

    grupos = [("baja_precision", "alta_precision", "precision"),
              ("baja_recall", "alta_recall", "recall")]
    rng = np.random.default_rng(0)

    fig, ax = plt.subplots(1, 2, figsize=(7.4, 3.0))

    pos, etiquetas, colores = [], [], []
    for i, (kb, ka, nom) in enumerate(grupos):
        for j, (k, col, dir_) in enumerate(((kb, AZUL, "reposo"),
                                            (ka, ROJO, "sobrecarga"))):
            v = vals(k)
            p = i * 2.4 + j
            ax[0].scatter(rng.normal(p, 0.08, len(v)), v, s=18, color=col,
                          alpha=0.7, edgecolor="black", linewidth=0.3, zorder=3)
            ax[0].hlines(np.mean(v), p - 0.3, p + 0.3, color="black", lw=1.8,
                         zorder=4)
            ax[0].text(p, 1.08, f"{np.mean(v):.2f}", ha="center", fontsize=8)
            pos.append(p); etiquetas.append(f"{dir_}\n{nom}"); colores.append(col)
    ax[0].set_xticks(pos)
    ax[0].set_xticklabels(etiquetas, fontsize=7)
    ax[0].set_ylim(-0.05, 1.18)
    ax[0].set_ylabel("valor por sujeto")
    ax[0].set_title("(a) El lazo es asimetrico")

    # (b) barrido del umbral de sobrecarga
    thetas = [1.5, 1.6, 1.7, 1.8, 1.9, 2.0]
    prec, rec, ndet = [], [], []
    for th in thetas:
        res = run_all(df, replace(base, theta_hi=th))
        s = summarise(res)
        prec.append(s["alta_precision"]["mean"])
        rec.append(s["alta_recall"]["mean"])
        ndet.append(sum(1 for v in res.values() if v["alta_precision"] is not None))
    ax[1].plot(thetas, prec, "o-", color=ROJO, lw=1.4, label="precision (alta)")
    ax[1].plot(thetas, rec, "s-", color=AZUL, lw=1.4, label="recall (alta)")
    ax[1].set_ylim(0, 1)
    ax[1].set_xlabel(r"umbral de sobrecarga $\theta_{hi}$")
    ax[1].set_ylabel("valor medio entre sujetos")
    ax[1].legend(loc="upper left", framealpha=0.9)
    ax[1].axvline(base.theta_hi, ls=":", c=GRIS, lw=1)
    axb = ax[1].twinx()
    axb.bar(thetas, ndet, width=0.06, color="#cccccc", zorder=0)
    axb.set_ylim(0, 15)
    axb.set_ylabel("sujetos donde se detecta (de 15)", fontsize=8)
    ax[1].set_zorder(axb.get_zorder() + 1)
    ax[1].patch.set_visible(False)
    ax[1].set_title("(b) Ningun umbral rescata la sobrecarga")

    fig.tight_layout()
    fig.savefig(OUT / "fig_lazo_asimetria.pdf")
    plt.close(fig)


def main():
    df = cargar()
    base = PolicyConfig()
    filas = ablacion(df, base)
    print(f"  {'guardas':22s}{'zona/min':>10}{'acc./min':>10}{'concord.':>10}")
    for n, sw, ac, ag in filas:
        print(f"  {n:22s}{sw['mean']:10.2f}{ac['mean']:10.2f}{ag['mean']:10.3f}")
    sid, per = fig_traza(df, base, filas)
    fig_asimetria(df, base, per)
    print(f"\n  sujeto ejemplo: {sid}")
    print("OK:", sorted(p.name for p in OUT.glob("*.pdf")))


if __name__ == "__main__":
    main()
