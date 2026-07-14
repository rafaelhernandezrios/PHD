"""
Deep-dive del pipeline sujeto-por-sujeto.

Para cada sujeto (data/Data-Experimento-Rafa/ + electron/):
  1. Carga CSV y estima SR real desde timestamps
  2. Para cada fase (baseline_open, baseline_closed, low, high):
     - preprocess (notch 60 + bandpass 1-40)
     - CAR
     - detección + interpolación de artefactos
     - PSD de Fz y Pz
     - bandpower theta(Fz) / alpha(Pz) por ventana (SR real, 1 s)
  3. Guarda figuras (pipeline, PSD, ratio ventaneado, resumen) y CSV por sujeto

Al final genera:
  - output/deep_dive_all/<subject>/figures/...
  - output/deep_dive_all/_master_summary.csv
  - output/deep_dive_all/_per_subject_ratios.png   (comparación)
  - output/deep_dive_all/_artifacts_per_subject.png
"""

from __future__ import annotations
import os, sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import signal, stats
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# -------- Config --------
BASE = Path(__file__).resolve().parents[2]
ROOTS = [BASE / "data" / "Data-Experimento-Rafa", BASE / "electron"]
OUT = BASE / "output" / "deep_dive_all"
OUT.mkdir(parents=True, exist_ok=True)

CH = {0: "Fp1", 1: "Fp2", 2: "F3", 3: "Fz", 4: "F4", 5: "P3", 6: "Pz", 7: "P4"}
FZ, PZ = 3, 6
THETA, ALPHA = (4.0, 7.0), (8.0, 12.0)
Z_TH, IQR_M, AMP_TH = 3.0, 3.0, 200.0
WIN_S = 1.0
PHASES = ["baseline_eyes_open", "baseline_eyes_closed",
          "low_cognitive_load", "high_cognitive_load"]
PCOLOR = {"baseline_eyes_open": "#6baed6", "baseline_eyes_closed": "#2171b5",
          "low_cognitive_load": "#fdae6b", "high_cognitive_load": "#d94801"}
LBL_MAP = {  # normalize CSV label→canonical
    "baseline_eyes_open": "baseline_eyes_open",
    "baseline_eyes_closed": "baseline_eyes_closed",
    "low_cognitive_load": "low_cognitive_load",
    "high_cognitive_load": "high_cognitive_load",
    "low_load": "low_cognitive_load",
    "high_load": "high_cognitive_load",
}


# -------- Pipeline helpers --------
def bandpass(x, sr, lo=1.0, hi=40.0, order=4):
    if sr < hi * 2 or len(x) < 3: return x
    nyq = sr / 2
    b, a = signal.butter(order, [lo / nyq, hi / nyq], btype="band")
    return signal.filtfilt(b, a, x)


def notch(x, sr, f0=60.0, Q=30.0):
    if sr < f0 * 2 or len(x) < 3: return x
    nyq = sr / 2
    b, a = signal.iirnotch(f0 / nyq, Q)
    return signal.filtfilt(b, a, x)


def prepro(x, sr):
    return bandpass(notch(x, sr), sr)


def car(eeg):
    return eeg - np.mean(eeg, axis=1, keepdims=True)


def detect_art(x):
    if len(x) < 3: return np.zeros(len(x), bool)
    mz = np.abs(stats.zscore(x)) > Z_TH
    q1, q3 = np.percentile(x, [25, 75])
    iqr = q3 - q1
    miq = (x < q1 - IQR_M * iqr) | (x > q3 + IQR_M * iqr)
    ma = np.abs(x) > AMP_TH
    return mz | miq | ma


def interp_art(x, mask):
    if mask.sum() == 0: return x.copy()
    out = x.copy()
    vi = np.where(~mask)[0]; ai = np.where(mask)[0]
    if len(vi) < 2: return out
    f = interp1d(vi, out[vi], kind="linear", fill_value="extrapolate", bounds_error=False)
    out[ai] = f(ai)
    return out


def bp_welch(x, band, sr):
    if sr <= 0 or len(x) < max(8, sr / 2): return 0.0
    nperseg = int(min(len(x), sr))
    if nperseg < 8: return 0.0
    freqs, psd = signal.welch(x, sr, nperseg=nperseg, noverlap=nperseg // 2)
    m = (freqs >= band[0]) & (freqs <= band[1])
    if not m.any(): return 0.0
    return float(np.trapezoid(psd[m], freqs[m]))


def windowed(fz, pz, sr, win_s=WIN_S):
    ws = max(int(round(win_s * sr)), 8)
    if len(fz) < ws or len(pz) < ws: return [], [], [], []
    step = max(ws // 2, 1)
    rs, ts, as_, starts = [], [], [], []
    for s in range(0, len(fz) - ws + 1, step):
        e = s + ws
        t = bp_welch(fz[s:e], THETA, sr)
        a = bp_welch(pz[s:e], ALPHA, sr)
        if a > 0 and np.isfinite(t) and np.isfinite(a):
            r = t / a
            if np.isfinite(r) and 0.01 < r < 100:
                rs.append(r); ts.append(t); as_.append(a); starts.append(s / sr)
    return rs, ts, as_, starts


# -------- Per-subject run --------
def estimate_sr(df):
    ts = pd.to_numeric(df["timestamp"], errors="coerce").dropna().sort_values()
    dt = ts.diff().dropna()
    dt = dt[dt > 0]
    if len(dt) == 0: return 250.0
    return float(1.0 / dt.median())


def run_subject(name: str, csv: Path, subject_out: Path, log):
    log(f"\n{'=' * 72}\nSUJETO: {name}\n{'=' * 72}")
    log(f"CSV: {csv}")
    df = pd.read_csv(csv)
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df = df.sort_values("timestamp").reset_index(drop=True)
    n_total = len(df)
    sr = estimate_sr(df)
    dur = df["timestamp"].iloc[-1] - df["timestamp"].iloc[0] if n_total > 1 else 0
    log(f"rows={n_total:,}  dur={dur:.1f}s  SR≈{sr:.2f} Hz")

    df["phase_can"] = df["label"].map(LBL_MAP)

    summary_rows = []
    per_phase = {}
    for ph in PHASES:
        sub = df[df["phase_can"] == ph].reset_index(drop=True)
        n = len(sub)
        if n == 0:
            log(f"  [{ph}] SIN DATOS"); continue
        d = sub["timestamp"].iloc[-1] - sub["timestamp"].iloc[0] if n > 1 else 0
        phsr = n / d if d > 0 else sr
        eeg = np.column_stack([sub[f"channel_{c}"].values for c in range(8)]).astype(float)
        eeg_f = np.zeros_like(eeg)
        for c in range(8):
            eeg_f[:, c] = prepro(eeg[:, c], phsr)
        eeg_c = car(eeg_f)
        fz_r, pz_r = eeg[:, FZ], eeg[:, PZ]
        fz_f, pz_f = eeg_f[:, FZ], eeg_f[:, PZ]
        fz_c, pz_c = eeg_c[:, FZ], eeg_c[:, PZ]
        fz_m, pz_m = detect_art(fz_c), detect_art(pz_c)
        fz_pct = 100 * fz_m.sum() / max(len(fz_c), 1)
        pz_pct = 100 * pz_m.sum() / max(len(pz_c), 1)
        fz_cl = interp_art(fz_c, fz_m)
        pz_cl = interp_art(pz_c, pz_m)
        rs, ts, as_, starts = windowed(fz_cl, pz_cl, phsr)
        mean_r = float(np.mean(rs)) if rs else np.nan
        med_r = float(np.median(rs)) if rs else np.nan
        std_r = float(np.std(rs)) if rs else np.nan
        mean_t = float(np.mean(ts)) if ts else np.nan
        mean_a = float(np.mean(as_)) if as_ else np.nan

        log(f"  [{ph}] n={n}  dur={d:.1f}s  sr={phsr:.1f}  artFz={fz_pct:.1f}%  artPz={pz_pct:.1f}%  "
            f"win={len(rs)}  ratio={mean_r:.3f}  θ={mean_t:.3f}  α={mean_a:.3f}")
        summary_rows.append(dict(
            subject=name, phase=ph, n_samples=n, duration_s=d, sr_hz=phsr,
            fz_artifact_pct=fz_pct, pz_artifact_pct=pz_pct,
            n_windows=len(rs), mean_ratio=mean_r, median_ratio=med_r, std_ratio=std_r,
            mean_theta_fz=mean_t, mean_alpha_pz=mean_a,
        ))
        per_phase[ph] = dict(
            n=n, dur=d, sr=phsr,
            fz_r=fz_r, pz_r=pz_r, fz_f=fz_f, pz_f=pz_f,
            fz_c=fz_c, pz_c=pz_c, fz_m=fz_m, pz_m=pz_m,
            fz_cl=fz_cl, pz_cl=pz_cl,
            fz_pct=fz_pct, pz_pct=pz_pct,
            ratios=np.asarray(rs), starts=np.asarray(starts),
            thetas=np.asarray(ts), alphas=np.asarray(as_),
            mean_r=mean_r,
        )

    fig_dir = subject_out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Plot A: raw timeline (8 channels, phases shaded)
    try:
        t_all = df["timestamp"].values - df["timestamp"].iloc[0]
        fig, axes = plt.subplots(8, 1, figsize=(14, 12), sharex=True)
        for c in range(8):
            axes[c].plot(t_all, df[f"channel_{c}"].values, "-", color="k", lw=0.3, alpha=0.7)
            for ph in PHASES:
                m = df["phase_can"] == ph
                if m.any():
                    axes[c].fill_between(t_all, 0, 1, where=m.values,
                                         transform=axes[c].get_xaxis_transform(),
                                         color=PCOLOR[ph], alpha=0.18, lw=0)
            axes[c].set_ylabel(CH[c], fontsize=9)
            axes[c].grid(alpha=0.2)
        axes[-1].set_xlabel("Time (s)")
        from matplotlib.patches import Patch
        axes[0].legend(handles=[Patch(color=PCOLOR[p], alpha=0.5, label=p) for p in PHASES],
                       loc="upper right", fontsize=7, ncol=4)
        fig.suptitle(f"{name} — raw 8-ch, SR≈{sr:.1f} Hz", fontsize=12)
        plt.tight_layout()
        plt.savefig(fig_dir / "01_raw_timeline.png", dpi=100); plt.close()
    except Exception as e:
        log(f"  ! fig01 error: {e}")

    # Plot B: pipeline per phase (5 rows x 2 cols)
    for ph, r in per_phase.items():
        try:
            sr_p = r["sr"]; t = np.arange(r["n"]) / sr_p
            fig, ax = plt.subplots(5, 2, figsize=(13, 13))
            fig.suptitle(f"{name} — {ph} (sr={sr_p:.1f}Hz, dur={r['dur']:.1f}s)", fontsize=12, fontweight="bold")
            for col, cn, raw, filt, cc, mk, cl in [
                (0, "Fz", r["fz_r"], r["fz_f"], r["fz_c"], r["fz_m"], r["fz_cl"]),
                (1, "Pz", r["pz_r"], r["pz_f"], r["pz_c"], r["pz_m"], r["pz_cl"]),
            ]:
                ax[0, col].plot(t, raw, color="k", lw=0.4); ax[0, col].set_title(f"{cn} raw"); ax[0, col].grid(alpha=0.3)
                ax[1, col].plot(t, filt, color="#4c72b0", lw=0.4); ax[1, col].set_title(f"{cn} notch+band"); ax[1, col].grid(alpha=0.3)
                ax[2, col].plot(t, cc, color="#55a868", lw=0.4)
                if mk.any():
                    ax[2, col].scatter(t[mk], cc[mk], s=3, color="#c44e52", zorder=3, label=f"art ({mk.sum()})")
                    ax[2, col].legend(fontsize=7)
                ax[2, col].set_title(f"{cn} CAR + art"); ax[2, col].grid(alpha=0.3)
                ax[3, col].plot(t, cl, color="#8172b2", lw=0.4); ax[3, col].set_title(f"{cn} cleaned"); ax[3, col].grid(alpha=0.3)
                if len(cl) >= max(8, sr_p / 2) and sr_p > 0:
                    nper = int(min(len(cl), sr_p * 4))
                    if nper >= 8:
                        fr, ps = signal.welch(cl, sr_p, nperseg=nper, noverlap=nper // 2)
                        ax[4, col].semilogy(fr, ps, color="#4c72b0")
                        ax[4, col].axvspan(*THETA, color="#c44e52", alpha=0.15, label="θ 4-7")
                        ax[4, col].axvspan(*ALPHA, color="#55a868", alpha=0.15, label="α 8-12")
                        ax[4, col].set_xlim(0, min(40, sr_p / 2 - 1))
                        ax[4, col].legend(fontsize=7)
                ax[4, col].set_title(f"{cn} PSD"); ax[4, col].set_xlabel("Hz"); ax[4, col].grid(alpha=0.3, which="both")
            plt.tight_layout()
            plt.savefig(fig_dir / f"02_pipeline_{ph.replace('_', '-')}.png", dpi=100); plt.close()
        except Exception as e:
            log(f"  ! fig02 {ph}: {e}")

    # Plot C: PSD overlay by phase
    try:
        fig, ax = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
        for j, (cn, key) in enumerate([("Fz", "fz_cl"), ("Pz", "pz_cl")]):
            for ph, r in per_phase.items():
                x = r[key]; sp = r["sr"]
                if len(x) < max(8, sp / 2) or sp <= 0: continue
                nper = int(min(len(x), sp * 4))
                if nper < 8: continue
                fr, ps = signal.welch(x, sp, nperseg=nper, noverlap=nper // 2)
                ax[j].semilogy(fr, ps, color=PCOLOR[ph], lw=1.2, label=ph)
            ax[j].axvspan(*THETA, color="#c44e52", alpha=0.10)
            ax[j].axvspan(*ALPHA, color="#55a868", alpha=0.10)
            ax[j].set_xlim(0, 25); ax[j].set_title(f"{cn} — PSD por fase")
            ax[j].set_xlabel("Hz"); ax[j].grid(alpha=0.3, which="both")
            ax[j].legend(fontsize=7)
            if j == 0: ax[j].set_ylabel("PSD [µV²/Hz]")
        fig.suptitle(f"{name} — PSD cleaned por fase", fontsize=12)
        plt.tight_layout()
        plt.savefig(fig_dir / "03_psd_overlay.png", dpi=100); plt.close()
    except Exception as e:
        log(f"  ! fig03: {e}")

    # Plot D: windowed ratios
    try:
        fig, (axs, axb) = plt.subplots(1, 2, figsize=(13, 4.5))
        for ph, r in per_phase.items():
            if len(r["ratios"]) == 0: continue
            axs.plot(r["starts"], r["ratios"], "-o", ms=3, color=PCOLOR[ph], alpha=0.7, label=ph)
        axs.set_yscale("log"); axs.set_xlabel("Time (s)"); axs.set_ylabel("θ(Fz)/α(Pz)")
        axs.set_title("Ratio ventaneado"); axs.grid(alpha=0.3, which="both"); axs.legend(fontsize=7)
        data = [per_phase[p]["ratios"] for p in PHASES if p in per_phase and len(per_phase[p]["ratios"]) > 0]
        labs = [p for p in PHASES if p in per_phase and len(per_phase[p]["ratios"]) > 0]
        if data:
            bp = axb.boxplot(data, tick_labels=labs, patch_artist=True, showmeans=True)
            for pp, lab in zip(bp["boxes"], labs):
                pp.set_facecolor(PCOLOR[lab]); pp.set_alpha(0.6)
            axb.set_yscale("log"); axb.tick_params(axis="x", rotation=20)
            axb.set_title("Distribución por fase"); axb.grid(alpha=0.3, which="both")
        fig.suptitle(f"{name} — ratios ventaneados", fontsize=12)
        plt.tight_layout()
        plt.savefig(fig_dir / "04_windowed_ratios.png", dpi=100); plt.close()
    except Exception as e:
        log(f"  ! fig04: {e}")

    # Plot E: summary bars
    try:
        phs = [p for p in PHASES if p in per_phase]
        vals = [per_phase[p]["mean_r"] for p in phs]
        fig, ax = plt.subplots(figsize=(8, 4.5))
        colors = [PCOLOR[p] for p in phs]
        ax.bar(range(len(phs)), [v if np.isfinite(v) else 0 for v in vals], color=colors, edgecolor="k")
        ax.set_xticks(range(len(phs))); ax.set_xticklabels(phs, rotation=15)
        ax.set_ylabel("Mean θ(Fz)/α(Pz)"); ax.set_title(f"{name} — mean ratio por fase (SR={sr:.1f}Hz)")
        ax.grid(axis="y", alpha=0.3)
        for i, v in enumerate(vals):
            if np.isfinite(v): ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
        plt.tight_layout()
        plt.savefig(fig_dir / "05_summary.png", dpi=100); plt.close()
    except Exception as e:
        log(f"  ! fig05: {e}")

    # Per-subject CSV
    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(subject_out / "summary.csv", index=False)

    return summary_rows, sr, n_total


# -------- Main --------
def main():
    log_path = OUT / "_log.txt"
    log_f = open(log_path, "w", encoding="utf-8")

    def log(*a):
        s = " ".join(str(x) for x in a)
        print(s); log_f.write(s + "\n"); log_f.flush()

    log("=" * 72); log("DEEP DIVE — TODOS LOS SUJETOS"); log("=" * 72)

    # Find all data_* folders
    seen = set()
    jobs = []
    for root in ROOTS:
        if not root.exists(): continue
        for d in sorted(root.glob("data_*")):
            name = d.name.replace("data_", "").strip()
            key = name.lower()
            if key in seen: continue
            csvs = sorted(d.glob("eeg_data_*.csv"))
            if not csvs: continue
            jobs.append((name, csvs[-1]))  # latest
            seen.add(key)

    log(f"Sujetos encontrados: {len(jobs)}")
    all_rows = []
    meta = []
    for name, csv in jobs:
        try:
            sub_out = OUT / name
            rows, sr, n_total = run_subject(name, csv, sub_out, log)
            all_rows.extend(rows)
            meta.append(dict(subject=name, sr=sr, n_total=n_total, n_phases=len(rows)))
        except Exception as e:
            log(f"! ERROR {name}: {e}")
            import traceback; traceback.print_exc()

    # Master CSV
    if all_rows:
        master = pd.DataFrame(all_rows)
        master.to_csv(OUT / "_master_summary.csv", index=False)
        log(f"\nSaved: {OUT / '_master_summary.csv'}  ({len(master)} rows)")

        # ------- Master comparison figure: ratios per phase per subject -------
        pivot = master.pivot_table(index="subject", columns="phase", values="mean_ratio", aggfunc="first")
        pivot = pivot.reindex(columns=PHASES)
        srs = master.groupby("subject")["sr_hz"].first()
        wins = master.groupby("subject")["n_windows"].sum()
        # Order by SR (desc) then by name
        order = srs.sort_values(ascending=False).index.tolist()
        pivot = pivot.reindex(order)
        wins = wins.reindex(order)

        fig, ax = plt.subplots(figsize=(max(14, len(order) * 0.9), 6))
        x = np.arange(len(order))
        w = 0.2
        for i, ph in enumerate(PHASES):
            vals = pivot[ph].values
            vals = [v if np.isfinite(v) else 0 for v in vals]
            ax.bar(x + (i - 1.5) * w, vals, w, label=ph, color=PCOLOR[ph], edgecolor="k", lw=0.3)
        ax.set_xticks(x)
        labels = [f"{s}\nSR={srs[s]:.1f}" for s in order]
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Mean θ(Fz)/α(Pz)")
        ax.set_title("Ratio theta(Fz)/alpha(Pz) por fase — todos los sujetos (SR real)")
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=8, ncol=4, loc="upper right")
        plt.tight_layout()
        plt.savefig(OUT / "_per_subject_ratios.png", dpi=120); plt.close()
        log(f"Saved: {OUT / '_per_subject_ratios.png'}")

        # ------- Master artifacts figure -------
        fz_piv = master.pivot_table(index="subject", columns="phase", values="fz_artifact_pct", aggfunc="first")
        pz_piv = master.pivot_table(index="subject", columns="phase", values="pz_artifact_pct", aggfunc="first")
        fz_piv = fz_piv.reindex(index=order, columns=PHASES)
        pz_piv = pz_piv.reindex(index=order, columns=PHASES)
        fig, axes = plt.subplots(2, 1, figsize=(max(14, len(order) * 0.9), 8))
        for axi, mat, lbl in [(axes[0], fz_piv, "Fz"), (axes[1], pz_piv, "Pz")]:
            for i, ph in enumerate(PHASES):
                vals = mat[ph].values
                vals = [v if np.isfinite(v) else 0 for v in vals]
                axi.bar(x + (i - 1.5) * w, vals, w, label=ph, color=PCOLOR[ph], edgecolor="k", lw=0.3)
            axi.axhline(50, color="red", ls="--", lw=1, alpha=0.5, label="50% umbral")
            axi.set_xticks(x)
            axi.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
            axi.set_ylabel(f"{lbl} % artefactos")
            axi.set_title(f"% artefactos detectados — canal {lbl}")
            axi.grid(axis="y", alpha=0.3)
            axi.legend(fontsize=7, ncol=5, loc="upper right")
        plt.tight_layout()
        plt.savefig(OUT / "_artifacts_per_subject.png", dpi=120); plt.close()
        log(f"Saved: {OUT / '_artifacts_per_subject.png'}")

        # ------- Berger + H>L sanity table -------
        sanity = []
        for s in order:
            r = master[master["subject"] == s]
            def g(p, col="mean_ratio"):
                row = r[r["phase"] == p]
                return row[col].iloc[0] if len(row) else np.nan
            a_open = g("baseline_eyes_open", "mean_alpha_pz")
            a_closed = g("baseline_eyes_closed", "mean_alpha_pz")
            lo = g("low_cognitive_load"); hi = g("high_cognitive_load")
            berger = (np.isfinite(a_open) and np.isfinite(a_closed) and a_closed > a_open)
            hl = (np.isfinite(lo) and np.isfinite(hi) and hi > lo)
            n_win_lh = r[r["phase"].isin(["low_cognitive_load", "high_cognitive_load"])]["n_windows"].min()
            sanity.append(dict(subject=s, sr_hz=srs[s],
                               low=lo, high=hi, delta=hi - lo if (np.isfinite(hi) and np.isfinite(lo)) else np.nan,
                               alpha_open=a_open, alpha_closed=a_closed,
                               berger_ok=berger, high_gt_low=hl,
                               min_windows_low_high=n_win_lh))
        san = pd.DataFrame(sanity)
        san.to_csv(OUT / "_sanity_table.csv", index=False)
        log(f"Saved: {OUT / '_sanity_table.csv'}")

        log("\nTabla sanity:")
        log(san.to_string(index=False))

    log_f.close()


if __name__ == "__main__":
    main()
