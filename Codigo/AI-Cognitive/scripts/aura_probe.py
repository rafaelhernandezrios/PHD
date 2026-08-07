"""Feasibility probe on the in-house AURA recordings.

Question: do the sessions that sample fast enough support the frontal-theta /
parietal-alpha story, and therefore the CLI and montage contributions that the
downloaded arithmetic/Stroop dataset cannot support (it has no parietal
channel)?

Only sessions whose median sampling interval puts alpha below Nyquist are
usable. At fs = 2.6 Hz, Nyquist is 1.3 Hz and both theta and alpha are
unrecoverable; those sessions are excluded here, not filtered.

Channel map is the one documented for this device in Cognitive-load/README.md,
and unlike the downloaded dataset it belongs to these recordings:
    ch0 Fp1  ch1 Fp2  ch2 F3  ch3 Fz  ch4 F4  ch5 P3  ch6 Pz  ch7 P4
"""
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt, welch

DATA = Path("/Users/rafael/Documents/Proyectos Personales/Doctorado/PHD/Codigo/"
            "Cognitive-load/data")
CH = {0: "Fp1", 1: "Fp2", 2: "F3", 3: "Fz", 4: "F4", 5: "P3", 6: "Pz", 7: "P4"}

FS_MIN = 30.0        # need Nyquist > 12 Hz to see alpha at all
FS_TARGET = 50.0
WIN_SEC, STRIDE_SEC = 4.0, 2.0
BANDS = {"theta": (4, 7), "alpha": (8, 12)}
PHASES = {"baseline_eyes_open": "baseline", "baseline_eyes_closed": "baseline",
          "low_load": "low", "high_load": "high"}


def load_session(path):
    d = pd.read_csv(path)
    t = pd.to_numeric(d["timestamp"], errors="coerce")
    ok = t.notna()
    d, t = d[ok], t[ok].to_numpy()
    if len(t) < 100 or t[-1] <= t[0]:
        return None
    fs = len(t) / (t[-1] - t[0])
    if fs < FS_MIN:
        return None, fs
    X = d[[f"channel_{i}" for i in range(8)]].apply(pd.to_numeric, errors="coerce")
    phase = d["phase"].map(PHASES)
    keep = X.notna().all(axis=1) & phase.notna()
    return (t[keep.to_numpy()], X[keep].to_numpy(), phase[keep].to_numpy()), fs


def resample_uniform(t, X, fs=FS_TARGET):
    """Irregular timestamps -> uniform grid, so Welch means what it says."""
    grid = np.arange(t[0], t[-1], 1.0 / fs)
    out = np.column_stack([interp1d(t, X[:, c], kind="linear")(grid)
                           for c in range(X.shape[1])])
    return grid, out


def band_powers(seg, fs):
    b, a = butter(4, [1.0 / (fs / 2), 20.0 / (fs / 2)], btype="band")
    out = {}
    for c, name in CH.items():
        x = filtfilt(b, a, seg[:, c] - seg[:, c].mean())
        f, P = welch(x, fs=fs, nperseg=min(len(x), int(fs * 2)))
        for bn, (lo, hi) in BANDS.items():
            out[f"{name}_{bn}"] = float(P[(f >= lo) & (f <= hi)].mean())
    return out


def main():
    files = sorted(glob.glob(str(DATA / "Data-Experimento-*/*/eeg_data_*.csv")))
    rows, used, skipped = [], [], []

    for f in files:
        subj = Path(f).parent.name
        res = load_session(f)
        if res is None:
            skipped.append((subj, float("nan"), "sin timestamps"))
            continue
        payload, fs = res
        if payload is None:
            skipped.append((subj, fs, f"fs {fs:.1f} Hz, Nyquist {fs/2:.1f} < 12"))
            continue
        t, X, phase = payload
        grid, Xu = resample_uniform(t, X)
        ph_u = pd.Series(phase).reindex(
            np.searchsorted(t, grid, side="right") - 1).to_numpy()

        n = int(WIN_SEC * FS_TARGET)
        step = int(STRIDE_SEC * FS_TARGET)
        for s in range(0, len(grid) - n, step):
            seg, lab = Xu[s:s + n], ph_u[s:s + n]
            if pd.isna(lab).any() or len(set(lab)) != 1:
                continue
            bp = band_powers(seg, FS_TARGET)
            bp.update(subject=subj, phase=lab[0])
            rows.append(bp)
        used.append((subj, fs))

    df = pd.DataFrame(rows)
    print(f"sesiones usables: {len(used)}   descartadas: {len(skipped)}")
    for s, fs in used:
        print(f"   OK       {s:16s} fs {fs:5.1f} Hz")
    for s, fs, why in skipped:
        print(f"   descarto {s:16s} {why}")

    if df.empty:
        print("\nsin ventanas: nada que evaluar")
        return

    print(f"\nventanas: {len(df)}   sujetos: {df.subject.nunique()}")
    print(f"por fase: {dict(df.phase.value_counts())}\n")

    # ---- la prueba fisiologica, normalizada al baseline de cada sujeto ----
    df["CLI"] = df["Fz_theta"] / df["Pz_alpha"]
    print("cambio relativo al baseline del propio sujeto (mediana entre sujetos):")
    print(f"   {'medida':16s}{'baseline':>10}{'low':>10}{'high':>10}   sube con la carga?")
    for measure, name in [("Fz_theta", "theta en Fz"),
                          ("Pz_alpha", "alpha en Pz"),
                          ("CLI", "CLI theta/alpha")]:
        vals = {}
        for ph in ("baseline", "low", "high"):
            per = []
            for s, g in df.groupby("subject"):
                base = g.loc[g.phase == "baseline", measure].median()
                cur = g.loc[g.phase == ph, measure].median()
                if base and np.isfinite(base) and np.isfinite(cur):
                    per.append(cur / base)
            vals[ph] = np.median(per) if per else float("nan")
        trend = "si" if vals["high"] > vals["baseline"] else "no"
        if measure == "Pz_alpha":
            trend = "baja: " + ("si" if vals["high"] < vals["baseline"] else "no")
        print(f"   {name:16s}{vals['baseline']:10.2f}{vals['low']:10.2f}"
              f"{vals['high']:10.2f}   {trend}")

    out = Path(__file__).resolve().parent.parent / "csv" / "aura_probe_features.csv"
    df.to_csv(out, index=False)
    print(f"\n[csv] {out}")


if __name__ == "__main__":
    main()

# ---------------------------------------------------------------------------
# Resultado de esta sonda (2026-07-31), para no repetirla:
#
#   7 de 16 sesiones muestrean lo bastante rapido para ver alfa. Ocho van a
#   2.6 Hz, con Nyquist en 1.3 Hz: theta y alfa son irrecuperables, no es un
#   problema de filtrado.
#
#   La firma fisiologica SI esta, normalizada al baseline de cada sujeto:
#       theta en Fz   1.00 -> 0.94 -> 1.93
#       alpha en Pz   1.00 -> 0.67 -> 0.95
#       CLI           1.00 -> 0.96 -> 1.30
#   Pero "low" no queda entre baseline y high, asi que no hay estructura
#   ordinal: Spearman del CLI contra el nivel = -0.069 +- 0.378.
#
#   Discriminabilidad por ventana, LOSO con calibracion, mismas 16 features
#   theta/alfa en ambos datasets:
#       AS (Nirabi, n=15)   binario 0.813
#       AURA      (n=7)     binario 0.638, por debajo de su propio baseline
#                           por mayoria (0.657)
#
#   El control importa: AS alcanza 0.813 con solo esas 16 features, casi lo
#   mismo que con las 218 del pipeline completo. El conjunto de features no
#   es el limitante, asi que el 0.638 de AURA es de los datos.
#
#   Conclusion: AURA no puede sustituir a AS como base del paper de SII. A lo
#   sumo sostiene una figura fisiologica suplementaria, donde Pz si es real.
# ---------------------------------------------------------------------------
