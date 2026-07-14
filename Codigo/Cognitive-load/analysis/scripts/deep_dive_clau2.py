"""
Deep-dive pipeline walk-through for subject "clau2".

From raw EEG CSV → filtered → CAR → artifact detection → interpolation →
PSD per band → windowed Theta(Fz)/Alpha(Pz) ratio per phase.

Produces one folder with step-by-step figures and numeric log, so we can
see EXACTLY what happens at every stage for each of the 4 phases
(baseline_eyes_open, baseline_eyes_closed, low_cognitive_load,
high_cognitive_load).

This uses the same parameters as step4_cognitive_load_cleaned.py EXCEPT:
  - Uses the subject's ACTUAL sample rate in the bandpower (Welch) call,
    not the hard-coded SAMPLE_RATE = 250 from step4.
    We print both side-by-side so the effect of that choice is visible.

Output: output/deep_dive/clau2/
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy import signal, stats
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

# -------------------------------------------------------------------------
# Paths / config
# -------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_CSV = BASE_DIR / "electron" / "data_clau2" / "eeg_data_20260417_182251.csv"
OUT_DIR = BASE_DIR / "output" / "deep_dive" / "clau2"
FIG_DIR = OUT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

CHANNEL_NAMES = {0: "Fp1", 1: "Fp2", 2: "F3", 3: "Fz",
                 4: "F4", 5: "P3", 6: "Pz", 7: "P4"}
FZ_CH = 3
PZ_CH = 6

# Phase label mapping (raw → canonical).
# The CSV stores canonical names already in the 'label' column; the 'phase'
# column uses shorter aliases. We build the map defensively.
PHASE_MAP = {
    "baseline_eyes_open":   "baseline_eyes_open",
    "baseline_eyes_closed": "baseline_eyes_closed",
    "low_cognitive_load":   "low_cognitive_load",
    "high_cognitive_load":  "high_cognitive_load",
    # fallbacks (in case some CSVs use the short form in label)
    "low_load":             "low_cognitive_load",
    "high_load":            "high_cognitive_load",
}
CANONICAL_PHASES = ["baseline_eyes_open", "baseline_eyes_closed",
                    "low_cognitive_load", "high_cognitive_load"]
PHASE_COLORS = {
    "baseline_eyes_open":   "#6baed6",
    "baseline_eyes_closed": "#2171b5",
    "low_cognitive_load":   "#fdae6b",
    "high_cognitive_load":  "#d94801",
}

# Step4 parameters (kept identical)
THETA_BAND = (4.0, 7.0)
ALPHA_BAND = (8.0, 12.0)
Z_TH = 3.0
IQR_MULT = 3.0
AMP_TH = 200.0  # uV
WINDOW_SAMPLES_STEP4 = 250   # step4 uses 250 samples (= 1 s @ 250 Hz)

# Logging
LOG_PATH = OUT_DIR / "log.txt"


class Tee:
    def __init__(self, p):
        self.f = open(p, "w", encoding="utf-8")

    def __call__(self, *a):
        s = " ".join(str(x) for x in a)
        print(s); self.f.write(s + "\n"); self.f.flush()

    def close(self):
        self.f.close()


log = Tee(LOG_PATH)


# -------------------------------------------------------------------------
# Pipeline helpers (same logic as step4 — copied to make this self-contained)
# -------------------------------------------------------------------------
def bandpass(x, sr, lo=1.0, hi=40.0, order=4):
    if sr < hi * 2 or len(x) < 3:
        return x
    nyq = sr / 2.0
    b, a = signal.butter(order, [lo / nyq, hi / nyq], btype="band")
    return signal.filtfilt(b, a, x)


def notch(x, sr, f0=60.0, Q=30.0):
    if sr < f0 * 2 or len(x) < 3:
        return x
    nyq = sr / 2.0
    b, a = signal.iirnotch(f0 / nyq, Q)
    return signal.filtfilt(b, a, x)


def preprocess(x, sr):
    return bandpass(notch(x, sr), sr)


def apply_car(eeg):
    car = np.mean(eeg, axis=1, keepdims=True)
    return eeg - car


def detect_artifacts(x, z_th=Z_TH, iqr_mult=IQR_MULT, amp_th=AMP_TH):
    """Combined artifact mask: z-score | IQR | amplitude."""
    if len(x) < 3:
        return np.zeros(len(x), dtype=bool)
    mask_z = np.abs(stats.zscore(x)) > z_th
    q1, q3 = np.percentile(x, [25, 75])
    iqr = q3 - q1
    mask_iqr = (x < q1 - iqr_mult * iqr) | (x > q3 + iqr_mult * iqr)
    mask_amp = np.abs(x) > amp_th
    return mask_z | mask_iqr | mask_amp


def interpolate_artifacts(x, mask):
    if mask.sum() == 0:
        return x.copy()
    out = x.copy()
    valid_idx = np.where(~mask)[0]
    art_idx = np.where(mask)[0]
    if len(valid_idx) < 2:
        return out
    f = interp1d(valid_idx, out[valid_idx], kind="linear",
                 fill_value="extrapolate", bounds_error=False)
    out[art_idx] = f(art_idx)
    return out


def bandpower_welch(x, band, sr):
    if len(x) < sr // 2:
        return 0.0
    nperseg = int(min(len(x), sr))
    freqs, psd = signal.welch(x, sr, nperseg=nperseg, noverlap=nperseg // 2)
    mask = (freqs >= band[0]) & (freqs <= band[1])
    if not mask.any():
        return 0.0
    return float(np.trapezoid(psd[mask], freqs[mask]))


def windowed_ratio(fz, pz, sr, win_samples):
    """Return (ratios, thetas, alphas, win_starts_in_sec)."""
    if len(fz) < win_samples or len(pz) < win_samples:
        return [], [], [], []
    step = win_samples // 2
    ratios, thetas, alphas, starts_s = [], [], [], []
    for start in range(0, len(fz) - win_samples + 1, step):
        end = start + win_samples
        t = bandpower_welch(fz[start:end], THETA_BAND, sr)
        a = bandpower_welch(pz[start:end], ALPHA_BAND, sr)
        if a > 0 and np.isfinite(t) and np.isfinite(a):
            r = t / a
            if np.isfinite(r) and 0.01 < r < 100:
                ratios.append(r); thetas.append(t); alphas.append(a)
                starts_s.append(start / sr)
    return ratios, thetas, alphas, starts_s


# -------------------------------------------------------------------------
# 1. Load raw data
# -------------------------------------------------------------------------
log("=" * 72)
log("DEEP DIVE — CLAU2")
log("=" * 72)
log(f"CSV: {DATA_CSV}")

df_all = pd.read_csv(DATA_CSV)
df_all["timestamp"] = pd.to_numeric(df_all["timestamp"], errors="coerce")
df_all = df_all.sort_values("timestamp").reset_index(drop=True)

# Estimate actual sample rate
time_diffs = df_all["timestamp"].diff().dropna()
median_interval = float(time_diffs.median())
actual_sr = 1.0 / median_interval if median_interval > 0 else 250.0
log(f"Total rows: {len(df_all):,}")
log(f"Total duration (s): {df_all['timestamp'].iloc[-1] - df_all['timestamp'].iloc[0]:.1f}")
log(f"Median interval: {median_interval*1000:.2f} ms → actual SR ≈ {actual_sr:.2f} Hz")

# step4 uses the 'label' column to segment (LABELS_OF_INTEREST = canonical names).
# Here the raw CSV uses 'low_load' / 'high_load' in label column.
# step4 filters on `df['label'] == label` with label from LABELS_OF_INTEREST —
# which means it would match nothing unless the CSV already uses canonical names.
# We normalize: take whatever is in `label`, map to canonical, and add a col.
# (This is also what the real pipeline must be doing via step1 or similar.)
raw_labels = df_all["label"].dropna().unique()
log(f"Raw labels in CSV: {sorted(raw_labels.tolist())}")

df_all["canonical_phase"] = df_all["label"].map(PHASE_MAP)
log("Mapped canonical phases (row counts):")
for ph in CANONICAL_PHASES:
    n = int((df_all["canonical_phase"] == ph).sum())
    log(f"  {ph:26s}  n={n:>6d}  dur≈{n/actual_sr:.1f}s")


# -------------------------------------------------------------------------
# 2. Global timeline plot — all channels, colored by phase
# -------------------------------------------------------------------------
def plot_global_timeline():
    fig, axes = plt.subplots(8, 1, figsize=(14, 14), sharex=True)
    t = df_all["timestamp"].values - df_all["timestamp"].iloc[0]

    for ch in range(8):
        ax = axes[ch]
        ax.plot(t, df_all[f"channel_{ch}"].values, "-", color="k", lw=0.4, alpha=0.7)
        # Shade canonical phases
        for ph in CANONICAL_PHASES:
            mask = df_all["canonical_phase"] == ph
            if mask.any():
                ax.fill_between(t, 0, 1, where=mask, transform=ax.get_xaxis_transform(),
                                color=PHASE_COLORS[ph], alpha=0.18, linewidth=0)
        ax.set_ylabel(CHANNEL_NAMES[ch], fontsize=10)
        ax.grid(alpha=0.2)
    axes[-1].set_xlabel("Time (s) since session start")
    # Legend
    from matplotlib.patches import Patch
    axes[0].legend(handles=[Patch(color=PHASE_COLORS[p], alpha=0.5, label=p) for p in CANONICAL_PHASES],
                   loc="upper right", fontsize=8, ncol=4)
    fig.suptitle("clau2 — raw EEG across all 8 channels, phases shaded", fontsize=13)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "01_raw_all_channels_timeline.png", dpi=120)
    plt.close()
    log("  Saved 01_raw_all_channels_timeline.png")

plot_global_timeline()


# -------------------------------------------------------------------------
# 3. Per-phase processing for Fz and Pz
# -------------------------------------------------------------------------
WINDOW_S = 1.0  # 1-second windows (matches step4 semantically)

results: Dict[str, Dict] = {}

for phase in CANONICAL_PHASES:
    log("\n" + "-" * 72)
    log(f"Phase: {phase}")
    sub = df_all[df_all["canonical_phase"] == phase].reset_index(drop=True)
    n = len(sub)
    if n == 0:
        log("  (no data)"); continue
    dur = float(sub["timestamp"].iloc[-1] - sub["timestamp"].iloc[0])
    phase_sr = n / dur if dur > 0 else actual_sr
    log(f"  rows={n:,}  dur={dur:.1f}s  sr_est={phase_sr:.1f} Hz")

    # 8-channel matrix
    eeg_raw = np.column_stack([sub[f"channel_{c}"].values for c in range(8)]).astype(float)

    # Preprocess each channel (notch + bandpass, at correct SR)
    eeg_filt = np.zeros_like(eeg_raw)
    for c in range(8):
        eeg_filt[:, c] = preprocess(eeg_raw[:, c], phase_sr)

    # CAR
    eeg_car = apply_car(eeg_filt)

    fz_raw = eeg_raw[:, FZ_CH]; pz_raw = eeg_raw[:, PZ_CH]
    fz_filt = eeg_filt[:, FZ_CH]; pz_filt = eeg_filt[:, PZ_CH]
    fz_car  = eeg_car[:, FZ_CH];  pz_car  = eeg_car[:, PZ_CH]

    # Artifact detection on CAR signals (as in step4)
    fz_mask = detect_artifacts(fz_car)
    pz_mask = detect_artifacts(pz_car)
    fz_art_pct = 100 * fz_mask.sum() / len(fz_car)
    pz_art_pct = 100 * pz_mask.sum() / len(pz_car)
    log(f"  artifacts — Fz: {fz_mask.sum()} ({fz_art_pct:.1f}%),  Pz: {pz_mask.sum()} ({pz_art_pct:.1f}%)")

    # Interpolation
    fz_clean = interpolate_artifacts(fz_car, fz_mask)
    pz_clean = interpolate_artifacts(pz_car, pz_mask)

    # ----- Windowed ratios, two ways -----
    # (a) Using actual SR (what we believe is correct)
    win_samples_real = max(int(WINDOW_S * phase_sr), 8)
    ratios_real, thetas_real, alphas_real, starts_real = windowed_ratio(
        fz_clean, pz_clean, phase_sr, win_samples_real)

    # (b) Using the step4 assumption: WINDOW_SAMPLES_STEP4=250 and sr=250
    ratios_step4, thetas_step4, alphas_step4, starts_step4 = windowed_ratio(
        fz_clean, pz_clean, 250.0, WINDOW_SAMPLES_STEP4)

    log(f"  windows (actual sr={phase_sr:.1f} Hz, win={win_samples_real} samples ≈ {WINDOW_S}s):")
    log(f"    valid ratios: {len(ratios_real)}")
    if ratios_real:
        log(f"    mean ratio = {np.mean(ratios_real):.4f}   median = {np.median(ratios_real):.4f}   std = {np.std(ratios_real):.4f}")
    log(f"  windows (step4 style sr=250, win=250 samples):")
    log(f"    valid ratios: {len(ratios_step4)}")
    if ratios_step4:
        log(f"    mean ratio = {np.mean(ratios_step4):.4f}   median = {np.median(ratios_step4):.4f}   std = {np.std(ratios_step4):.4f}")

    results[phase] = dict(
        n=n, dur=dur, sr=phase_sr,
        fz_raw=fz_raw, pz_raw=pz_raw,
        fz_filt=fz_filt, pz_filt=pz_filt,
        fz_car=fz_car, pz_car=pz_car,
        fz_mask=fz_mask, pz_mask=pz_mask,
        fz_art_pct=fz_art_pct, pz_art_pct=pz_art_pct,
        fz_clean=fz_clean, pz_clean=pz_clean,
        ratios_real=np.asarray(ratios_real),
        thetas_real=np.asarray(thetas_real),
        alphas_real=np.asarray(alphas_real),
        starts_real=np.asarray(starts_real),
        ratios_step4=np.asarray(ratios_step4),
        thetas_step4=np.asarray(thetas_step4),
        alphas_step4=np.asarray(alphas_step4),
        starts_step4=np.asarray(starts_step4),
    )


# -------------------------------------------------------------------------
# 4. Figures — per-phase walk-through
# -------------------------------------------------------------------------
def plot_step_by_step_per_phase():
    phases = [p for p in CANONICAL_PHASES if p in results]
    for ph in phases:
        r = results[ph]
        sr = r["sr"]
        t = np.arange(r["n"]) / sr

        fig, axes = plt.subplots(5, 2, figsize=(14, 14), sharex=False)
        fig.suptitle(f"clau2 — {ph} (sr≈{sr:.1f} Hz, dur={r['dur']:.1f} s)",
                     fontsize=13, fontweight="bold")

        # Column 0 = Fz, Column 1 = Pz
        for col, ch_name, raw, filt, car, mask, clean in [
            (0, "Fz", r["fz_raw"], r["fz_filt"], r["fz_car"], r["fz_mask"], r["fz_clean"]),
            (1, "Pz", r["pz_raw"], r["pz_filt"], r["pz_car"], r["pz_mask"], r["pz_clean"]),
        ]:
            # Row 0: raw
            ax = axes[0, col]
            ax.plot(t, raw, "-", color="k", lw=0.4)
            ax.set_title(f"{ch_name} — raw")
            ax.set_ylabel("µV")
            ax.grid(alpha=0.3)

            # Row 1: notch+bandpass
            ax = axes[1, col]
            ax.plot(t, filt, "-", color="#4c72b0", lw=0.4)
            ax.set_title(f"{ch_name} — after notch(60) + bandpass(1–40 Hz)")
            ax.set_ylabel("µV")
            ax.grid(alpha=0.3)

            # Row 2: CAR + artifact mask overlay
            ax = axes[2, col]
            ax.plot(t, car, "-", color="#55a868", lw=0.4, label="CAR")
            if mask.any():
                ax.scatter(t[mask], car[mask], s=4, color="#c44e52",
                           label=f"artifacts ({mask.sum()})", zorder=3)
            ax.set_title(f"{ch_name} — CAR + artifact flags")
            ax.set_ylabel("µV")
            ax.legend(loc="upper right", fontsize=8)
            ax.grid(alpha=0.3)

            # Row 3: cleaned (interpolated)
            ax = axes[3, col]
            ax.plot(t, clean, "-", color="#8172b2", lw=0.4)
            ax.set_title(f"{ch_name} — cleaned (interpolated)")
            ax.set_ylabel("µV")
            ax.grid(alpha=0.3)

            # Row 4: PSD (Welch on full phase, cleaned)
            ax = axes[4, col]
            if len(clean) >= sr // 2:
                nperseg = int(min(len(clean), sr * 4))
                freqs, psd = signal.welch(clean, sr, nperseg=nperseg, noverlap=nperseg // 2)
                ax.semilogy(freqs, psd, color="#4c72b0", lw=1.2)
                ax.axvspan(*THETA_BAND, color="#c44e52", alpha=0.15, label="Theta 4–7")
                ax.axvspan(*ALPHA_BAND, color="#55a868", alpha=0.15, label="Alpha 8–12")
            ax.set_title(f"{ch_name} — PSD (cleaned, Welch)")
            ax.set_xlabel("Hz")
            ax.set_ylabel("PSD [µV²/Hz]")
            ax.set_xlim(0, min(40, sr / 2 - 1))
            ax.legend(loc="upper right", fontsize=8)
            ax.grid(alpha=0.3, which="both")

        for row_ax in axes[:-1]:
            for ax in row_ax:
                ax.set_xlabel("Time (s)" if row_ax is axes[-2] else "")
        plt.tight_layout()
        safe = ph.replace("_", "-")
        plt.savefig(FIG_DIR / f"02_pipeline_{safe}.png", dpi=120)
        plt.close()
        log(f"  Saved 02_pipeline_{safe}.png")

plot_step_by_step_per_phase()


# -------------------------------------------------------------------------
# 5. PSD overlay — the 4 phases on top of each other (Fz and Pz)
# -------------------------------------------------------------------------
def plot_psd_overlay():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for j, (ax, ch_name, key_clean) in enumerate(
        [(axes[0], "Fz", "fz_clean"), (axes[1], "Pz", "pz_clean")]
    ):
        for ph, r in results.items():
            sr = r["sr"]; x = r[key_clean]
            if len(x) < sr // 2:
                continue
            nperseg = int(min(len(x), sr * 4))
            freqs, psd = signal.welch(x, sr, nperseg=nperseg, noverlap=nperseg // 2)
            ax.semilogy(freqs, psd, "-", color=PHASE_COLORS[ph], lw=1.3, label=ph)
        ax.axvspan(*THETA_BAND, color="#c44e52", alpha=0.10)
        ax.axvspan(*ALPHA_BAND, color="#55a868", alpha=0.10)
        ax.set_xlim(0, 25)
        ax.set_title(f"{ch_name} — PSD by phase")
        ax.set_xlabel("Hz")
        if j == 0: ax.set_ylabel("PSD [µV²/Hz]")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(alpha=0.3, which="both")
    fig.suptitle("clau2 — PSD of cleaned signals (Fz & Pz) across 4 phases", fontsize=13)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "03_psd_by_phase_overlay.png", dpi=120)
    plt.close()
    log("  Saved 03_psd_by_phase_overlay.png")

plot_psd_overlay()


# -------------------------------------------------------------------------
# 6. Per-window Theta/Alpha ratio across the 4 phases, both SR modes
# -------------------------------------------------------------------------
def plot_windowed_ratios():
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharey=False)
    for row, (mode_key, mode_label) in enumerate([
        ("real",  f"Actual SR per phase (~{actual_sr:.0f} Hz), 1-s windows"),
        ("step4", "Step4 assumption: SR=250, WINDOW=250 samples"),
    ]):
        ax_series = axes[row, 0]
        ax_box    = axes[row, 1]
        for ph, r in results.items():
            ratios = r[f"ratios_{mode_key}"]
            starts = r[f"starts_{mode_key}"]
            if len(ratios) == 0:
                continue
            ax_series.plot(starts, ratios, "-o", ms=3, color=PHASE_COLORS[ph], alpha=0.7, label=ph)
        ax_series.set_title(f"Per-window Theta(Fz)/Alpha(Pz) — {mode_label}")
        ax_series.set_xlabel("Time within phase (s)")
        ax_series.set_ylabel("Ratio")
        ax_series.set_yscale("log")
        ax_series.grid(alpha=0.3, which="both")
        ax_series.legend(loc="upper right", fontsize=8)

        # Box by phase
        data = [results[ph][f"ratios_{mode_key}"] for ph in CANONICAL_PHASES if ph in results
                and len(results[ph][f"ratios_{mode_key}"]) > 0]
        labels = [ph for ph in CANONICAL_PHASES if ph in results
                  and len(results[ph][f"ratios_{mode_key}"]) > 0]
        if data:
            bp = ax_box.boxplot(data, tick_labels=labels, patch_artist=True, showmeans=True)
            for p, lab in zip(bp["boxes"], labels):
                p.set_facecolor(PHASE_COLORS[lab]); p.set_alpha(0.6)
            ax_box.set_yscale("log")
            ax_box.set_title(f"Distribution — {mode_label}")
            ax_box.tick_params(axis="x", rotation=20)
            ax_box.grid(alpha=0.3, which="both")
    fig.suptitle("clau2 — windowed Theta/Alpha ratio per phase (two SR modes)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "04_windowed_ratios.png", dpi=120)
    plt.close()
    log("  Saved 04_windowed_ratios.png")

plot_windowed_ratios()


# -------------------------------------------------------------------------
# 7. Summary bar chart — mean ratio by phase, both modes
# -------------------------------------------------------------------------
def plot_summary_bars():
    fig, ax = plt.subplots(1, 1, figsize=(9, 5))
    phases = [p for p in CANONICAL_PHASES if p in results]
    x = np.arange(len(phases))
    w = 0.35
    means_real  = [float(np.mean(results[p]["ratios_real"]))  if len(results[p]["ratios_real"])  else np.nan for p in phases]
    means_step4 = [float(np.mean(results[p]["ratios_step4"])) if len(results[p]["ratios_step4"]) else np.nan for p in phases]
    ax.bar(x - w/2, means_real,  w, label=f"Actual SR (~{actual_sr:.0f} Hz)", color="#4c72b0", alpha=0.85, edgecolor="k")
    ax.bar(x + w/2, means_step4, w, label="Step4 (SR=250)",                   color="#c44e52", alpha=0.85, edgecolor="k")
    ax.set_xticks(x); ax.set_xticklabels(phases, rotation=15)
    ax.set_ylabel("Mean Theta/Alpha ratio")
    ax.set_title("clau2 — mean ratio per phase (actual SR vs step4 assumption)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    for xi, mr, ms in zip(x, means_real, means_step4):
        if np.isfinite(mr): ax.text(xi - w/2, mr, f"{mr:.2f}", ha="center", va="bottom", fontsize=9)
        if np.isfinite(ms): ax.text(xi + w/2, ms, f"{ms:.2f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "05_mean_ratio_summary.png", dpi=120)
    plt.close()
    log("  Saved 05_mean_ratio_summary.png")

plot_summary_bars()


# -------------------------------------------------------------------------
# 8. Theta(Fz) and Alpha(Pz) absolute powers per phase
# -------------------------------------------------------------------------
def plot_abs_powers():
    phases = [p for p in CANONICAL_PHASES if p in results]
    x = np.arange(len(phases))
    w = 0.35

    theta_real = [float(np.mean(results[p]["thetas_real"])) if len(results[p]["thetas_real"]) else np.nan for p in phases]
    alpha_real = [float(np.mean(results[p]["alphas_real"])) if len(results[p]["alphas_real"]) else np.nan for p in phases]
    theta_step = [float(np.mean(results[p]["thetas_step4"])) if len(results[p]["thetas_step4"]) else np.nan for p in phases]
    alpha_step = [float(np.mean(results[p]["alphas_step4"])) if len(results[p]["alphas_step4"]) else np.nan for p in phases]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, thetas, alphas, title in [
        (axes[0], theta_real, alpha_real, f"Actual SR (~{actual_sr:.0f} Hz)"),
        (axes[1], theta_step, alpha_step, "Step4 (SR=250)"),
    ]:
        ax.bar(x - w/2, thetas, w, label="Theta(Fz)", color="#c44e52", edgecolor="k", alpha=0.85)
        ax.bar(x + w/2, alphas, w, label="Alpha(Pz)", color="#55a868", edgecolor="k", alpha=0.85)
        ax.set_xticks(x); ax.set_xticklabels(phases, rotation=15)
        ax.set_yscale("log")
        ax.set_title(f"Absolute band powers — {title}")
        ax.legend()
        ax.grid(axis="y", alpha=0.3, which="both")
    fig.suptitle("clau2 — Theta(Fz) and Alpha(Pz) power per phase", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "06_band_powers.png", dpi=120)
    plt.close()
    log("  Saved 06_band_powers.png")

plot_abs_powers()


# -------------------------------------------------------------------------
# 9. Dump per-window CSV for reproducibility
# -------------------------------------------------------------------------
def save_window_csv():
    rows = []
    for ph, r in results.items():
        for mode in ["real", "step4"]:
            ratios = r[f"ratios_{mode}"]
            thetas = r[f"thetas_{mode}"]
            alphas = r[f"alphas_{mode}"]
            starts = r[f"starts_{mode}"]
            for i, (ra, th, al, st) in enumerate(zip(ratios, thetas, alphas, starts)):
                rows.append({"phase": ph, "mode": mode, "window_idx": i,
                             "start_s": float(st), "theta_fz": float(th),
                             "alpha_pz": float(al), "ratio": float(ra)})
    pd.DataFrame(rows).to_csv(OUT_DIR / "clau2_windows.csv", index=False)

    # Summary
    summary_rows = []
    for ph in CANONICAL_PHASES:
        if ph not in results: continue
        r = results[ph]
        for mode in ["real", "step4"]:
            ratios = r[f"ratios_{mode}"]
            thetas = r[f"thetas_{mode}"]
            alphas = r[f"alphas_{mode}"]
            summary_rows.append({
                "phase": ph, "mode": mode,
                "n_windows": len(ratios),
                "sr_used_Hz": r["sr"] if mode == "real" else 250.0,
                "mean_theta_fz": float(np.mean(thetas)) if len(thetas) else np.nan,
                "mean_alpha_pz": float(np.mean(alphas)) if len(alphas) else np.nan,
                "mean_ratio": float(np.mean(ratios)) if len(ratios) else np.nan,
                "median_ratio": float(np.median(ratios)) if len(ratios) else np.nan,
                "std_ratio": float(np.std(ratios)) if len(ratios) else np.nan,
                "fz_artifact_pct": r["fz_art_pct"],
                "pz_artifact_pct": r["pz_art_pct"],
            })
    pd.DataFrame(summary_rows).to_csv(OUT_DIR / "clau2_summary.csv", index=False)
    log("  Saved clau2_windows.csv + clau2_summary.csv")

save_window_csv()


# -------------------------------------------------------------------------
# Final summary print
# -------------------------------------------------------------------------
log("\n" + "=" * 72)
log("FINAL SUMMARY — mean ratio per phase")
log("=" * 72)
log(f"{'phase':28s}  {'mode':7s}  {'n_win':>6s}  {'sr_used':>7s}  {'theta':>9s}  {'alpha':>9s}  {'mean_ratio':>10s}")
for ph in CANONICAL_PHASES:
    if ph not in results: continue
    r = results[ph]
    for mode in ["real", "step4"]:
        ratios = r[f"ratios_{mode}"]
        thetas = r[f"thetas_{mode}"]
        alphas = r[f"alphas_{mode}"]
        sr_used = r["sr"] if mode == "real" else 250.0
        log(f"{ph:28s}  {mode:7s}  {len(ratios):>6d}  {sr_used:>7.1f}  "
            f"{(np.mean(thetas) if len(thetas) else np.nan):>9.3f}  "
            f"{(np.mean(alphas) if len(alphas) else np.nan):>9.3f}  "
            f"{(np.mean(ratios) if len(ratios) else np.nan):>10.4f}")

log("\nDONE.")
log.close()
