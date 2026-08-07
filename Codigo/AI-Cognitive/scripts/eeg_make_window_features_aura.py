"""Cleaned window-feature extraction for the AURA-based AS dataset.

KEY CHANGE vs. eeg_make_window_features.py:
  - Only v1..v8 are real EEG channels (the 8 AURA electrodes).
  - v0 is a sample counter, v9-v11 are accelerometer, v12-v21 are device
    metadata, v22 is unix timestamp, v23 is zero.
  - We rename v1..v8 to their documented 10-20 positions:
      v1=Fp1, v2=Fp2, v3=F3, v4=Fz, v5=F4, v6=P3, v7=Pz, v8=P4.
  - All non-EEG columns are dropped before any feature computation.

Output: csv/eeg_window_features_aura.csv
"""
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.signal import welch
from scipy.stats import entropy

from eeg_make_window_features import (
    FS,
    bandpower,
    preprocess_for_bandpower,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WINDOW_SIZE = int(4 * FS)
STEP_SIZE = int(2 * FS)

# Montaje del dataset publico de aritmetica/Stroop:
#   Nirabi et al., Data in Brief 60:111477 (2025), doi 10.1016/j.dib.2025.111477
#   Mendeley Data 10.17632/kt38js3jv7.1, CC BY.
# OpenBCI Cyton de 8 canales a 250 Hz. NO tiene electrodos parietales.
#
# El mapeo anterior era Fp1,Fp2,F3,Fz,F4,P3,Pz,P4: el de la diadema AURA propia
# del laboratorio, documentado en Cognitive-load/README.md, que se aplicaba por
# error a estas grabaciones descargadas. Con aquel mapeo lo que se llamaba Fz
# era en realidad F3 y lo que se llamaba Pz era F8.
#
# SUPUESTO PENDIENTE DE VERIFICAR: que las columnas v1..v8 aparecen en el orden
# en que el articulo lista los electrodos. Comprobarlo contra la documentacion
# del dataset en Mendeley antes de publicar cualquier afirmacion que dependa de
# la identidad de un canal concreto.
CHANNELS = {
    "v1": "Fp1", "v2": "Fp2", "v3": "F7", "v4": "F3",
    "v5": "Fz",  "v6": "F4",  "v7": "F8", "v8": "C4",
}
EEG_COLS_RAW = list(CHANNELS.keys())            # v1..v8
EEG_COLS = [CHANNELS[c] for c in EEG_COLS_RAW]  # Fp1..C4

# Subconjuntos topograficos. Este montaje es frontal salvo por C4, asi que el
# contraste frontal-parietal de la literatura no es computable aqui.
FRONTAL = ["F7", "F3", "Fz", "F4", "F8"]
CENTRAL = ["C4"]
LR_PAIRS = [("Fp1", "Fp2"), ("F3", "F4"), ("F7", "F8")]


def compute_window_features(segment: np.ndarray) -> Dict[str, float]:
    """segment shape: (n_samples, 8) ordered as EEG_COLS."""
    feats: Dict[str, float] = {}
    bands = {
        "delta": (1.0, 4.0), "theta": (4.0, 8.0),
        "alpha": (8.0, 13.0), "beta": (13.0, 30.0), "gamma": (30.0, 40.0),
    }
    bp_by_channel: Dict[str, Dict[str, float]] = {}
    for i, ch in enumerate(EEG_COLS):
        x_raw = segment[:, i]
        x = preprocess_for_bandpower(x_raw, FS)
        # time-domain
        feats[f"{ch}_mean"] = float(np.mean(x))
        feats[f"{ch}_std"] = float(np.std(x))
        feats[f"{ch}_min"] = float(np.min(x))
        feats[f"{ch}_max"] = float(np.max(x))
        feats[f"{ch}_rms"] = float(np.sqrt(np.mean(x ** 2)))
        feats[f"{ch}_abs_mean"] = float(np.mean(np.abs(x)))
        # band powers
        bp = {n: bandpower(x, FS, b) for n, b in bands.items()}
        bp_by_channel[ch] = bp
        for n, p in bp.items():
            feats[f"{ch}_bp_{n}"] = float(p)
        total = float(sum(bp.values()) + 1e-12)
        feats[f"{ch}_bp_total"] = total
        for n, p in bp.items():
            feats[f"{ch}_rel_bp_{n}"] = float(p / total)
        a = bp["alpha"] if bp["alpha"] > 0 else 1e-9
        b = bp["beta"]  if bp["beta"]  > 0 else 1e-9
        t = bp["theta"] if bp["theta"] > 0 else 1e-9
        feats[f"{ch}_theta_alpha_ratio"] = float(bp["theta"] / a)
        feats[f"{ch}_beta_alpha_ratio"]  = float(bp["beta"] / a)
        feats[f"{ch}_theta_beta_ratio"]  = float(t / b)
        feats[f"{ch}_engagement"] = float(bp["beta"] / (bp["alpha"] + bp["theta"] + 1e-12))
        feats[f"{ch}_load_index"] = float(t / a)
        # spectral entropy
        freqs, psd = welch(x, FS, nperseg=min(len(x), int(4 * FS)))
        idx = np.logical_and(freqs >= 1.0, freqs <= 40.0)
        pn = psd[idx] / (psd[idx].sum() + 1e-12)
        feats[f"{ch}_spec_entropy"] = float(entropy(pn))
        # Hjorth
        dx = np.diff(x); ddx = np.diff(dx)
        vx = float(np.var(x)) + 1e-12
        vd = float(np.var(dx)) + 1e-12
        vdd = float(np.var(ddx)) + 1e-12
        mob = float(np.sqrt(vd / vx))
        feats[f"{ch}_hjorth_activity"]  = vx
        feats[f"{ch}_hjorth_mobility"]  = mob
        feats[f"{ch}_hjorth_complexity"] = float(np.sqrt(vdd / vd) / (mob + 1e-12))

    # ---- Cross-channel features now meaningful (real montage) ----
    # Hemispheric asymmetry (log) on alpha and beta for documented L/R pairs
    for left, right in LR_PAIRS:
        a_l = bp_by_channel[left]["alpha"] + 1e-9
        a_r = bp_by_channel[right]["alpha"] + 1e-9
        b_l = bp_by_channel[left]["beta"]  + 1e-9
        b_r = bp_by_channel[right]["beta"] + 1e-9
        feats[f"asym_alpha_{left}_{right}"] = float(np.log(a_r) - np.log(a_l))
        feats[f"asym_beta_{left}_{right}"]  = float(np.log(b_r) - np.log(b_l))

    # Analogo del indice de carga. El clasico es Theta(Fz)/Alpha(Pz), pero este
    # montaje no tiene Pz; C4 es el unico sitio no frontal disponible, asi que
    # el cociente NO es el indice de la literatura y se nombra en consecuencia.
    theta_fz = bp_by_channel["Fz"]["theta"]
    alpha_c4 = bp_by_channel["C4"]["alpha"] + 1e-9
    feats["theta_Fz_over_alpha_C4"] = float(theta_fz / alpha_c4)

    # Contraste topografico frontal vs central, por la misma razon
    for band in ("theta", "alpha", "beta"):
        f_mean = np.mean([bp_by_channel[c][band] for c in FRONTAL])
        c_mean = np.mean([bp_by_channel[c][band] for c in CENTRAL])
        feats[f"frontal_mean_{band}"] = float(f_mean)
        feats[f"central_mean_{band}"] = float(c_mean)
        feats[f"front_minus_central_{band}"] = float(f_mean - c_mean)

    return feats


def main() -> None:
    in_csv = PROJECT_ROOT / "csv" / "eeg_all_samples.csv"
    df = pd.read_csv(in_csv)
    # Drop rows with unknown condition
    df = df[df["condition_4"] != "unknown"].reset_index(drop=True)
    # Rename v1..v8 to EEG names; drop all non-EEG columns
    rename = {raw: name for raw, name in CHANNELS.items()}
    df = df.rename(columns=rename)

    group_cols = ["file_name", "subject_id", "task_type",
                  "condition_4", "load_3", "load_2"]
    all_feats: List[Dict] = []
    for _, g in df.groupby(group_cols, sort=False):
        g = g.reset_index(drop=True)
        data = g[EEG_COLS].to_numpy(dtype=np.float64)
        n = len(g)
        start = 0
        while start + WINDOW_SIZE <= n:
            end = start + WINDOW_SIZE
            feats = compute_window_features(data[start:end, :])
            feats["file_name"]   = g.loc[0, "file_name"]
            feats["subject_id"]  = g.loc[0, "subject_id"]
            feats["task_type"]   = g.loc[0, "task_type"]
            feats["condition_4"] = g.loc[0, "condition_4"]
            feats["load_3"]      = g.loc[0, "load_3"]
            feats["load_2"]      = g.loc[0, "load_2"]
            feats["window_start_idx"] = int(start)
            feats["window_end_idx"]   = int(end)
            all_feats.append(feats)
            start += STEP_SIZE

    feats_df = pd.DataFrame(all_feats)
    out = PROJECT_ROOT / "csv" / "eeg_window_features_aura.csv"
    feats_df.to_csv(out, index=False)
    print(f"Wrote {len(feats_df)} rows × {feats_df.shape[1]} cols → {out}")
    print(f"EEG channels used: {EEG_COLS}")


if __name__ == "__main__":
    main()
