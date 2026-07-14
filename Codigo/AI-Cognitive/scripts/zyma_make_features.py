"""Build window-features CSV for PhysioNet Zyma (eegmat 1.0.0) dataset.

For each subject:
  - Subject{NN}_1.edf : rest baseline (eyes closed, ~3 min)  → load_2=normal
  - Subject{NN}_2.edf : mental subtraction task (~1 min)      → load_2=alta
"""
from pathlib import Path
import re
import numpy as np
import pandas as pd
import mne

from eeg_make_window_features import (
    FS as DEFAULT_FS,  # we override per-file from EDF
    WINDOW_SIZE as _UNUSED_WINDOW,
    bandpower, preprocess_for_bandpower,
)
from scipy.signal import welch
from scipy.stats import entropy

ROOT = Path(__file__).resolve().parents[1]
EDF_DIR = ROOT / "raw_data" / "Zyma"
OUT_CSV = ROOT / "csv" / "zyma_window_features.csv"

WIN_SEC = 4.0
STEP_SEC = 2.0
TARGET_FS = 250.0  # resample to match the AS pipeline


def list_edf():
    files = sorted(EDF_DIR.glob("Subject*_*.edf"))
    out = []
    for f in files:
        m = re.match(r"Subject(\d+)_(\d)\.edf", f.name)
        if not m:
            continue
        out.append((int(m.group(1)), int(m.group(2)), f))
    return out


def features_for_window(seg, ch_names, fs):
    feats = {}
    bands = {
        "delta": (1.0, 4.0), "theta": (4.0, 8.0),
        "alpha": (8.0, 13.0), "beta": (13.0, 30.0), "gamma": (30.0, 40.0),
    }
    for i, ch in enumerate(ch_names):
        x_raw = seg[:, i]
        x = preprocess_for_bandpower(x_raw, fs)
        feats[f"{ch}_mean"] = float(np.mean(x))
        feats[f"{ch}_std"] = float(np.std(x))
        feats[f"{ch}_min"] = float(np.min(x))
        feats[f"{ch}_max"] = float(np.max(x))
        feats[f"{ch}_rms"] = float(np.sqrt(np.mean(x ** 2)))
        feats[f"{ch}_abs_mean"] = float(np.mean(np.abs(x)))
        bp = {n: bandpower(x, fs, b) for n, b in bands.items()}
        for n, p in bp.items():
            feats[f"{ch}_bp_{n}"] = float(p)
        total = float(sum(bp.values()) + 1e-12)
        feats[f"{ch}_bp_total"] = total
        for n, p in bp.items():
            feats[f"{ch}_rel_bp_{n}"] = float(p / total)
        a = bp["alpha"] if bp["alpha"] > 0 else 1e-9
        b = bp["beta"] if bp["beta"] > 0 else 1e-9
        t = bp["theta"] if bp["theta"] > 0 else 1e-9
        feats[f"{ch}_theta_alpha_ratio"] = float(bp["theta"] / a)
        feats[f"{ch}_beta_alpha_ratio"] = float(bp["beta"] / a)
        feats[f"{ch}_theta_beta_ratio"] = float(t / b)
        feats[f"{ch}_engagement"] = float(bp["beta"] / (bp["alpha"] + bp["theta"] + 1e-12))
        feats[f"{ch}_load_index"] = float(t / a)
        freqs, psd = welch(x, fs, nperseg=min(len(x), int(WIN_SEC * fs)))
        idx = np.logical_and(freqs >= 1.0, freqs <= 40.0)
        pn = psd[idx] / (psd[idx].sum() + 1e-12)
        feats[f"{ch}_spec_entropy"] = float(entropy(pn))
        dx = np.diff(x); ddx = np.diff(dx)
        vx = float(np.var(x)) + 1e-12
        vd = float(np.var(dx)) + 1e-12
        vdd = float(np.var(ddx)) + 1e-12
        mob = float(np.sqrt(vd / vx))
        feats[f"{ch}_hjorth_activity"] = vx
        feats[f"{ch}_hjorth_mobility"] = mob
        feats[f"{ch}_hjorth_complexity"] = float(np.sqrt(vdd / vd) / (mob + 1e-12))
    return feats


def process_file(subj, sess, path):
    raw = mne.io.read_raw_edf(path, preload=True, verbose="ERROR")
    # keep only EEG channels (drop ECG, A2-A1 reference)
    eeg_picks = [c for c in raw.ch_names if c.startswith("EEG ") and "A2-A1" not in c]
    raw.pick(eeg_picks)
    # resample to target fs
    if raw.info["sfreq"] != TARGET_FS:
        raw.resample(TARGET_FS, npad="auto", verbose="ERROR")
    data = raw.get_data().T  # shape (n_samples, n_channels)
    ch_names = [c.replace("EEG ", "").replace(" ", "_") for c in raw.ch_names]
    fs = TARGET_FS
    win = int(WIN_SEC * fs)
    step = int(STEP_SEC * fs)
    # label: session 1 = baseline rest, session 2 = arithmetic task
    if sess == 1:
        load_2 = "normal"; load_3 = "normal"; cond = "natural"
    else:
        load_2 = "alta"; load_3 = "high"; cond = "highlevel"
    rows = []
    start = 0
    while start + win <= len(data):
        seg = data[start:start + win, :]
        f = features_for_window(seg, ch_names, fs)
        f["file_name"] = f"Subject{subj:02d}_{sess}.edf"
        f["subject_id"] = subj
        f["task_type"] = "Arithmetic"
        f["condition_4"] = cond
        f["load_2"] = load_2
        f["load_3"] = load_3
        f["window_start_idx"] = start
        f["window_end_idx"] = start + win
        rows.append(f)
        start += step
    return rows


def main():
    files = list_edf()
    print(f"Found {len(files)} EDF files")
    all_rows = []
    for subj, sess, path in files:
        rows = process_file(subj, sess, path)
        all_rows.extend(rows)
        print(f"  subj {subj:02d} sess {sess}: {len(rows)} windows")
    df = pd.DataFrame(all_rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nWrote {len(df)} rows × {df.shape[1]} cols → {OUT_CSV}")
    print("Label distribution (load_2):")
    print(df.load_2.value_counts())


if __name__ == "__main__":
    main()
