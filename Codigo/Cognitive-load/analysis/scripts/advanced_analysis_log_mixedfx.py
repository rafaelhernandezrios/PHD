"""
Advanced analysis: log-transform + within-subject baseline normalization +
mixed-effects at window level.

Changes vs deep_dive_all.py:
  1. Log-transform: log_ratio = log(theta_Fz) - log(alpha_Pz)
  2. Within-subject normalization: subtract each subject's baseline_eyes_open mean
  3. Window-level mixed-effects model (phase fixed, subject random)
  4. Also reports classic subject-mean Wilcoxon for comparison

Usable subjects only (SR >= 45 Hz, Fz/Pz artifacts < 30%).
Outputs:
  output/advanced_log_mixed/
    _windows.csv                  -- per-window log ratios (all subjects)
    _subject_means.csv            -- per-subject-phase aggregates
    _mixed_effects.txt            -- MLM output
    _paired_tests.csv             -- Wilcoxon classic vs normalized
    _figure_comparison.png        -- ratios raw vs log vs log-normalized
    _log.txt
"""

from __future__ import annotations
import os, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import signal, stats
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parents[2]
ROOTS = [BASE / "data" / "Data-Experimento-Rafa", BASE / "electron"]
OUT = BASE / "output" / "advanced_log_mixed"
OUT.mkdir(parents=True, exist_ok=True)

CH = {0: "Fp1", 1: "Fp2", 2: "F3", 3: "Fz", 4: "F4", 5: "P3", 6: "Pz", 7: "P4"}
FZ, PZ = 3, 6
THETA, ALPHA = (4.0, 7.0), (8.0, 12.0)
Z_TH, IQR_M, AMP_TH = 3.0, 3.0, 200.0
WIN_S = 2.0            # 2 s windows -> 0.5 Hz resolution
MIN_SR = 45.0          # require near-50 Hz data
MAX_ART = 30.0         # drop subject/phase with Fz or Pz artifacts > 30%
PHASES = ["baseline_eyes_open", "baseline_eyes_closed",
          "low_cognitive_load", "high_cognitive_load"]
LBL_MAP = {
    "baseline_eyes_open": "baseline_eyes_open",
    "baseline_eyes_closed": "baseline_eyes_closed",
    "low_cognitive_load": "low_cognitive_load",
    "high_cognitive_load": "high_cognitive_load",
    "low_load": "low_cognitive_load",
    "high_load": "high_cognitive_load",
}


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


def bp_welch(x, band, sr, nperseg_s=2.0):
    if sr <= 0 or len(x) < 8: return 0.0
    nperseg = int(min(len(x), nperseg_s * sr))
    if nperseg < 8: return 0.0
    freqs, psd = signal.welch(x, sr, nperseg=nperseg, noverlap=nperseg // 2)
    m = (freqs >= band[0]) & (freqs <= band[1])
    if not m.any(): return 0.0
    return float(np.trapezoid(psd[m], freqs[m]))


def estimate_sr(df):
    ts = pd.to_numeric(df["timestamp"], errors="coerce").dropna().sort_values()
    dt = ts.diff().dropna()
    dt = dt[dt > 0]
    if len(dt) == 0: return 250.0
    return float(1.0 / dt.median())


def windowed_logs(fz, pz, sr, win_s=WIN_S):
    ws = max(int(round(win_s * sr)), 8)
    if len(fz) < ws or len(pz) < ws: return []
    step = max(ws // 2, 1)
    rows = []
    for s in range(0, len(fz) - ws + 1, step):
        e = s + ws
        t = bp_welch(fz[s:e], THETA, sr, nperseg_s=win_s)
        a = bp_welch(pz[s:e], ALPHA, sr, nperseg_s=win_s)
        if t > 0 and a > 0 and np.isfinite(t) and np.isfinite(a):
            rows.append((s / sr, t, a, np.log(t), np.log(a), np.log(t) - np.log(a)))
    return rows


def process_subject(name: str, csv: Path, log):
    log(f"\n{'=' * 72}\nSUJETO: {name}")
    df = pd.read_csv(csv)
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df = df.sort_values("timestamp").reset_index(drop=True)
    sr = estimate_sr(df)
    log(f"  CSV: {csv.name}  rows={len(df):,}  SR≈{sr:.2f} Hz")

    if sr < MIN_SR:
        log(f"  -> EXCLUIDO (SR<{MIN_SR})"); return None

    df["phase_can"] = df["label"].map(LBL_MAP)

    rows_all = []
    per_phase_art = {}
    for ph in PHASES:
        sub = df[df["phase_can"] == ph].reset_index(drop=True)
        n = len(sub)
        if n < 100: log(f"  [{ph}] <100 muestras, saltado"); continue
        d = sub["timestamp"].iloc[-1] - sub["timestamp"].iloc[0] if n > 1 else 0
        phsr = n / d if d > 0 else sr
        eeg = np.column_stack([sub[f"channel_{c}"].values for c in range(8)]).astype(float)
        eeg_f = np.zeros_like(eeg)
        for c in range(8):
            eeg_f[:, c] = prepro(eeg[:, c], phsr)
        eeg_c = car(eeg_f)
        fz, pz = eeg_c[:, FZ], eeg_c[:, PZ]
        fz_m, pz_m = detect_art(fz), detect_art(pz)
        fz_pct = 100 * fz_m.sum() / max(len(fz), 1)
        pz_pct = 100 * pz_m.sum() / max(len(pz), 1)
        per_phase_art[ph] = (fz_pct, pz_pct)
        if fz_pct > MAX_ART or pz_pct > MAX_ART:
            log(f"  [{ph}] art alto (Fz={fz_pct:.1f}% Pz={pz_pct:.1f}%), fase saltada")
            continue
        fz_cl = interp_art(fz, fz_m)
        pz_cl = interp_art(pz, pz_m)
        wins = windowed_logs(fz_cl, pz_cl, phsr)
        log(f"  [{ph}] n={n}  sr={phsr:.1f}  artFz={fz_pct:.1f}%  artPz={pz_pct:.1f}%  windows={len(wins)}")
        for (t_start, th, al, lth, lal, lr) in wins:
            rows_all.append((name, ph, t_start, th, al, lth, lal, lr))
    if not rows_all: return None
    return pd.DataFrame(rows_all, columns=[
        "subject", "phase", "t_start", "theta_fz", "alpha_pz",
        "log_theta", "log_alpha", "log_ratio",
    ])


def main():
    log_lines = []
    def L(s=""): print(s); log_lines.append(s)

    # find subjects (same logic as deep_dive_all)
    candidates = []
    seen = set()
    for root in ROOTS:
        if not root.exists(): continue
        for folder in sorted(root.glob("data_*")):
            name = folder.name.replace("data_", "")
            if name in seen: continue
            csvs = sorted(folder.glob("*.csv"))
            if not csvs: continue
            candidates.append((name, csvs[-1]))
            seen.add(name)

    L(f"Candidatos: {len(candidates)}")
    dfs = []
    for name, csv in candidates:
        df = process_subject(name, csv, L)
        if df is not None and len(df) > 0: dfs.append(df)
    if not dfs:
        L("SIN DATOS"); return

    W = pd.concat(dfs, ignore_index=True)
    W.to_csv(OUT / "_windows.csv", index=False)
    L(f"\nVentanas totales: {len(W):,}  sujetos: {W['subject'].nunique()}  fases: {W['phase'].nunique()}")

    # ---- Within-subject normalization vs baseline_eyes_open ----
    base_mean = (W[W["phase"] == "baseline_eyes_open"]
                 .groupby("subject")["log_ratio"].mean()
                 .rename("base_logratio"))
    W = W.merge(base_mean, on="subject", how="left")
    W["log_ratio_norm"] = W["log_ratio"] - W["base_logratio"]

    # Subject means per phase (raw ratio, log ratio, normalized)
    W["ratio"] = W["theta_fz"] / W["alpha_pz"]
    sm = (W.groupby(["subject", "phase"])
          .agg(n_windows=("log_ratio", "size"),
               mean_ratio=("ratio", "mean"),
               median_ratio=("ratio", "median"),
               mean_logratio=("log_ratio", "mean"),
               mean_logratio_norm=("log_ratio_norm", "mean"))
          .reset_index())
    sm.to_csv(OUT / "_subject_means.csv", index=False)
    L("\nResumen por sujeto/fase (first 20):")
    L(sm.head(20).to_string(index=False))

    # ---- Paired Wilcoxon on subject means: High vs Low ----
    def paired_wilcoxon(col):
        pivot = sm.pivot_table(index="subject", columns="phase", values=col)
        pivot = pivot.dropna(subset=["low_cognitive_load", "high_cognitive_load"])
        lo = pivot["low_cognitive_load"].values
        hi = pivot["high_cognitive_load"].values
        diff = hi - lo
        n = len(diff)
        if n < 3:
            return dict(n=n, mean_diff=np.nan, median_diff=np.nan, wilcoxon_W=np.nan, p=np.nan, dz=np.nan)
        dz = diff.mean() / diff.std(ddof=1) if diff.std(ddof=1) > 0 else np.nan
        try:
            res = stats.wilcoxon(hi, lo, zero_method="wilcox", alternative="two-sided")
            W_stat, p = float(res.statistic), float(res.pvalue)
        except Exception:
            W_stat, p = np.nan, np.nan
        return dict(n=n, mean_diff=float(diff.mean()), median_diff=float(np.median(diff)),
                    wilcoxon_W=W_stat, p=p, dz=float(dz))

    rows = []
    for col in ["mean_ratio", "mean_logratio", "mean_logratio_norm"]:
        d = paired_wilcoxon(col); d["metric"] = col
        rows.append(d)
    paired = pd.DataFrame(rows)[["metric", "n", "mean_diff", "median_diff", "dz", "wilcoxon_W", "p"]]
    paired.to_csv(OUT / "_paired_tests.csv", index=False)
    L("\n--- Paired Wilcoxon High vs Low (subject-level) ---")
    L(paired.to_string(index=False))

    # ---- Mixed-effects at window level ----
    W2 = W.copy()
    W2["phase"] = pd.Categorical(W2["phase"],
                                 categories=["low_cognitive_load", "high_cognitive_load",
                                             "baseline_eyes_open", "baseline_eyes_closed"],
                                 ordered=False)

    def fit_mlm(dv, data):
        try:
            md = smf.mixedlm(f"{dv} ~ C(phase, Treatment(reference='low_cognitive_load'))",
                             data, groups=data["subject"],
                             re_formula="~1")
            mdf = md.fit(method="lbfgs", reml=True)
            return mdf
        except Exception as e:
            return str(e)

    L("\n--- Mixed-Effects Model (log_ratio, window-level) ---")
    mlm1 = fit_mlm("log_ratio", W2)
    mlm1_txt = mlm1.summary().as_text() if hasattr(mlm1, "summary") else str(mlm1)
    L(mlm1_txt)

    L("\n--- Mixed-Effects Model (log_ratio_norm, window-level) ---")
    W2n = W2.dropna(subset=["log_ratio_norm"]).reset_index(drop=True)
    mlm2 = fit_mlm("log_ratio_norm", W2n)
    mlm2_txt = mlm2.summary().as_text() if hasattr(mlm2, "summary") else str(mlm2)
    L(mlm2_txt)

    with open(OUT / "_mixed_effects.txt", "w") as f:
        f.write("Mixed-effects: log_ratio ~ phase + (1|subject)\n" + "=" * 70 + "\n")
        f.write(mlm1_txt + "\n\n")
        f.write("Mixed-effects: log_ratio_norm ~ phase + (1|subject)\n" + "=" * 70 + "\n")
        f.write(mlm2_txt + "\n")

    # ---- Figure: 3 panels (raw ratio / log ratio / normalized) bar + per-subject overlays ----
    order = ["baseline_eyes_open", "baseline_eyes_closed", "low_cognitive_load", "high_cognitive_load"]
    short = {"baseline_eyes_open": "base_open", "baseline_eyes_closed": "base_closed",
             "low_cognitive_load": "low", "high_cognitive_load": "high"}
    colors = ["#6baed6", "#2171b5", "#fdae6b", "#d94801"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    for ax, metric, title in [
        (axes[0], "mean_ratio", "θ/α (raw)"),
        (axes[1], "mean_logratio", "log(θ/α)"),
        (axes[2], "mean_logratio_norm", "log(θ/α) − base_open"),
    ]:
        piv = sm.pivot_table(index="subject", columns="phase", values=metric).reindex(columns=order)
        x = np.arange(len(order))
        means = piv.mean().values
        sems = piv.sem().values
        ax.bar(x, means, yerr=sems, color=colors, alpha=0.75, capsize=6, edgecolor="k", lw=0.6)
        # Overlay per-subject lines
        for _, row in piv.iterrows():
            ax.plot(x, row.values, "o-", color="gray", lw=0.7, ms=3, alpha=0.55)
        ax.set_xticks(x); ax.set_xticklabels([short[p] for p in order], rotation=15)
        ax.set_title(f"{title}  (N={piv.dropna().shape[0]})")
        ax.grid(alpha=0.25)
        if metric == "mean_logratio_norm": ax.axhline(0, color="k", lw=0.8)
    fig.suptitle("Comparación de métricas — subject-level (líneas grises = sujetos individuales)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(OUT / "_figure_comparison.png", dpi=130)
    plt.close()

    # ---- Figure: distribution of window log_ratio_norm, phase colored ----
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for i, ph in enumerate(order):
        vals = W["log_ratio_norm"][W["phase"] == ph].values
        if len(vals) == 0: continue
        parts = ax.violinplot([vals], positions=[i], widths=0.75, showmeans=True, showmedians=False)
        for pc in parts["bodies"]:
            pc.set_facecolor(colors[i]); pc.set_alpha(0.6); pc.set_edgecolor("k")
    ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.set_xticks(range(len(order))); ax.set_xticklabels([short[p] for p in order])
    ax.set_ylabel("log(θ/α) normalizado (− baseline_open)")
    ax.set_title("Distribución window-level del log-ratio normalizado (todos los sujetos pooled)")
    ax.grid(alpha=0.25)
    plt.tight_layout()
    fig.savefig(OUT / "_figure_window_violin.png", dpi=130)
    plt.close()

    # save log
    with open(OUT / "_log.txt", "w") as f:
        f.write("\n".join(str(x) for x in log_lines))

    L(f"\nSalidas en {OUT}")


if __name__ == "__main__":
    main()
