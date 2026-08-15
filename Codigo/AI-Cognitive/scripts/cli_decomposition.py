"""Que mitad del cociente theta(Fz)/alpha(Pz) hace el trabajo.

El paper afirma que el numerador esta plano y que el denominador es el que se
mueve. zyma_montage_cli.py calcula eso al vuelo pero el JSON que hay en disco
se genero antes de que ese bloque existiera, asi que las cifras no estaban
guardadas en ninguna parte. Este script las recalcula y las persiste para que
la tesis pueda citarlas.

Se corre sobre los dos datasets:

  - eegmat (19 canales 10-20): Fz y Pz son electrodos reales, asi que el CLI
    clasico es computable y se descompone directamente.
  - AS (OpenBCI de 8 canales, Fp1 Fp2 F7 F3 Fz F4 F8 C4): montaje frontal sin
    parietales. Ahi no hay CLI que descomponer; lo que se mide es si theta
    frontal sube y si alpha baja en cada canal disponible.

Salida: csv/cli_decomposition.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parent.parent


def ratio_task_over_rest(df, col, rest_mask, task_mask):
    """Mediana por sujeto de tarea/reposo, con Wilcoxon contra 1."""
    ratios = []
    for _, g in df.groupby("subject_id"):
        b = g.loc[rest_mask.loc[g.index], col].median()
        t = g.loc[task_mask.loc[g.index], col].median()
        if np.isfinite(b) and np.isfinite(t) and b > 0:
            ratios.append(t / b)
    ratios = np.asarray(ratios, dtype=float)
    w, p = wilcoxon(ratios, np.ones_like(ratios))
    return {
        "median_task_over_rest": float(np.median(ratios)),
        "n_up": int(np.sum(ratios > 1)),
        "n_subjects": int(len(ratios)),
        "wilcoxon_W": float(w),
        "wilcoxon_p": float(p),
        "ratios": [float(r) for r in ratios],
    }


def do_eegmat():
    df = pd.read_csv(ROOT / "csv" / "zyma_window_features.csv")
    rest = df.load_2 == "normal"
    task = df.load_2 == "alta"
    df["CLI"] = df["Fz_bp_theta"] / (df["Pz_bp_alpha"] + 1e-12)

    out = {}
    for name, col in (("CLI", "CLI"),
                      ("theta_Fz", "Fz_bp_theta"),
                      ("alpha_Pz", "Pz_bp_alpha")):
        out[name] = ratio_task_over_rest(df, col, rest, task)
        r = out[name]
        print(f"  eegmat {name:10s} tarea/reposo {r['median_task_over_rest']:.3f}  "
              f"sube en {r['n_up']}/{r['n_subjects']}  "
              f"W={r['wilcoxon_W']:.0f}  p={r['wilcoxon_p']:.3g}")

    # Cuanto del ascenso del CLI explica cada mitad por separado.
    out["_alpha_alone_would_give"] = 1.0 / out["alpha_Pz"]["median_task_over_rest"]
    out["_theta_alone_would_give"] = out["theta_Fz"]["median_task_over_rest"]
    print(f"  eegmat  solo la caida de alpha daria "
          f"{out['_alpha_alone_would_give']:.2f}x  "
          f"(observado {out['CLI']['median_task_over_rest']:.2f}x)")
    return out


def do_as():
    df = pd.read_csv(ROOT / "csv" / "eeg_window_features_aura.csv")
    rest = df.condition_4 == "normal"
    task = df.condition_4.isin(["low", "mid", "high"])
    chans = ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "C4"]

    out = {"_montage": chans, "_note": "montaje frontal, sin electrodos parietales"}
    for band in ("theta", "alpha"):
        out[band] = {}
        for ch in chans:
            col = f"{ch}_bp_{band}"
            if col not in df.columns:
                continue
            out[band][ch] = ratio_task_over_rest(df, col, rest, task)
            r = out[band][ch]
            print(f"  AS {band:5s} {ch:4s} tarea/reposo {r['median_task_over_rest']:.3f}  "
                  f"sube en {r['n_up']}/{r['n_subjects']}  p={r['wilcoxon_p']:.3g}")
    return out


def main():
    print("eegmat (19 ch, Fz y Pz reales)")
    res = {"eegmat": do_eegmat()}
    print("\nAS (OpenBCI 8 ch, montaje frontal)")
    res["arithmetic_stroop"] = do_as()

    out = ROOT / "csv" / "cli_decomposition.json"
    out.write_text(json.dumps(res, indent=2))
    print(f"\n[json] {out}")


if __name__ == "__main__":
    main()
