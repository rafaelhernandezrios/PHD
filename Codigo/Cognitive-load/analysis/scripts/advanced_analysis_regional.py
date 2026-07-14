"""
Advanced analysis — REGIONAL version.

Frontal theta = promedio log-power de F3, Fz, F4 (canales 2, 3, 4)
Parietal alpha = promedio log-power de P3, Pz, P4 (canales 5, 6, 7)

Sobre eso: log-ratio = log(theta_region) - log(alpha_region),
normalización intra-sujeto vs baseline_eyes_open,
modelo mixto de efectos aleatorios window-level.

Outputs:
  output/advanced_regional/
    _windows.csv
    _subject_means.csv
    _paired_tests.csv
    _mixed_effects.txt
    _figure_comparison.png
    _figure_window_violin.png
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
OUT = BASE / "output" / "advanced_regional"
OUT.mkdir(parents=True, exist_ok=True)

CH = {0: "Fp1", 1: "Fp2", 2: "F3", 3: "Fz", 4: "F4", 5: "P3", 6: "Pz", 7: "P4"}
FRONTAL = [2, 3, 4]   # F3, Fz, F4
PARIETAL = [5, 6, 7]  # P3, Pz, P4
THETA, ALPHA = (4.0, 7.0), (8.0, 12.0)
Z_TH, IQR_M, AMP_TH = 3.0, 3.0, 200.0
WIN_S = 2.0
MIN_SR = 45.0
MAX_ART_REGION = 30.0  # region-average artifact pct threshold

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


def prepro(x, sr): return bandpass(notch(x, sr), sr)


def car(eeg): return eeg - np.mean(eeg, axis=1, keepdims=True)


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


def regional_logpower(x_list, band, sr, win_s):
    """Mean of log power across a list of channel arrays, per window."""
    logs = []
    for x in x_list:
        p = bp_welch(x, band, sr, nperseg_s=win_s)
        if p > 0 and np.isfinite(p):
            logs.append(np.log(p))
    if not logs: return np.nan
    return float(np.mean(logs))


def windowed_regional(frontal_arrays, parietal_arrays, sr, win_s=WIN_S):
    """Returns list of (t_start, log_theta_region, log_alpha_region, log_ratio)."""
    L0 = min(len(a) for a in frontal_arrays + parietal_arrays)
    ws = max(int(round(win_s * sr)), 8)
    if L0 < ws: return []
    step = max(ws // 2, 1)
    rows = []
    for s in range(0, L0 - ws + 1, step):
        e = s + ws
        lth = regional_logpower([a[s:e] for a in frontal_arrays], THETA, sr, win_s)
        lal = regional_logpower([a[s:e] for a in parietal_arrays], ALPHA, sr, win_s)
        if np.isfinite(lth) and np.isfinite(lal):
            rows.append((s / sr, lth, lal, lth - lal))
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

        frontal, parietal = [], []
        art_front, art_par = [], []
        for c in FRONTAL:
            x = eeg_c[:, c]; m = detect_art(x)
            art_front.append(100 * m.sum() / max(len(x), 1))
            frontal.append(interp_art(x, m))
        for c in PARIETAL:
            x = eeg_c[:, c]; m = detect_art(x)
            art_par.append(100 * m.sum() / max(len(x), 1))
            parietal.append(interp_art(x, m))

        art_f_mean = np.mean(art_front); art_p_mean = np.mean(art_par)
        if art_f_mean > MAX_ART_REGION or art_p_mean > MAX_ART_REGION:
            log(f"  [{ph}] art region alto (F={art_f_mean:.1f}% P={art_p_mean:.1f}%), fase saltada")
            continue
        wins = windowed_regional(frontal, parietal, phsr)
        log(f"  [{ph}] n={n}  sr={phsr:.1f}  artF={art_f_mean:.1f}%  artP={art_p_mean:.1f}%  windows={len(wins)}")
        for (t_start, lth, lal, lr) in wins:
            rows_all.append((name, ph, t_start, lth, lal, lr))
    if not rows_all: return None
    return pd.DataFrame(rows_all, columns=[
        "subject", "phase", "t_start",
        "log_theta_region", "log_alpha_region", "log_ratio",
    ])


THESIS_FIG = (
    Path(__file__).resolve().parents[3]
    / "Propuesta_de_Tesis_Rafa"
    / "Capitulo4"
    / "figures"
    / "fig_regional_mlm_forest.png"
)

PHASE_CONTRASTS = [
    ("baseline_eyes_closed", "BL closed – Low"),
    ("baseline_eyes_open", "BL open – Low"),
    ("high_cognitive_load", "High – Low"),
]


def plot_mlm_forest(mlm, out_path: Path, title: str = "Mixed-effects model: phase contrasts (regional)"):
    """Horizontal forest plot of phase contrasts vs. Low (reference)."""
    prefix = "C(phase, Treatment(reference='low_cognitive_load'))[T."
    betas, lo, hi, labels = [], [], [], []
    for phase_key, label in PHASE_CONTRASTS:
        name = f"{prefix}{phase_key}]"
        betas.append(float(mlm.params[name]))
        ci = mlm.conf_int().loc[name]
        lo.append(float(ci[0]))
        hi.append(float(ci[1]))
        labels.append(label)

    y = np.arange(len(labels))
    fig_h = max(3.2, 0.9 * len(labels) + 1.8)
    fig, ax = plt.subplots(figsize=(8.5, fig_h))

    ax.errorbar(betas, y, xerr=[np.array(betas) - np.array(lo), np.array(hi) - np.array(betas)],
                fmt="o", color="#2171b5", ecolor="#2171b5", capsize=4, ms=7, lw=1.6)
    ax.axvline(0, color="gray", ls="--", lw=1)

    x_pad = 0.06
    x_text = max(hi) + x_pad
    for yi, b, l, h, lab in zip(y, betas, lo, hi, labels):
        sign = "+" if b >= 0 else ""
        ax.text(x_text, yi,
                f"β = {sign}{b:.3f}  [{sign if l >= 0 else ''}{l:.3f}, {sign if h >= 0 else ''}{h:.3f}]",
                va="center", ha="left", fontsize=9)

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel(r"$\beta$ ($\Delta$ log-ratio vs. Low)")
    ax.set_title(title, pad=12)
    ax.set_ylim(len(labels) - 0.45, -0.55)
    ax.set_xlim(min(lo) - 0.12, x_text + 0.55)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    log_lines = []
    def L(s=""): print(s); log_lines.append(str(s))

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

    # within-subject normalization vs baseline_eyes_open
    base_mean = (W[W["phase"] == "baseline_eyes_open"]
                 .groupby("subject")["log_ratio"].mean()
                 .rename("base_logratio"))
    W = W.merge(base_mean, on="subject", how="left")
    W["log_ratio_norm"] = W["log_ratio"] - W["base_logratio"]

    # regional ratio = exp(log_ratio) for comparison with raw
    W["ratio"] = np.exp(W["log_ratio"])

    sm = (W.groupby(["subject", "phase"])
          .agg(n_windows=("log_ratio", "size"),
               mean_ratio=("ratio", "mean"),
               median_ratio=("ratio", "median"),
               mean_logratio=("log_ratio", "mean"),
               mean_logratio_norm=("log_ratio_norm", "mean"))
          .reset_index())
    sm.to_csv(OUT / "_subject_means.csv", index=False)
    L("\nResumen por sujeto/fase:")
    L(sm.to_string(index=False))

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
    L("\n--- Paired Wilcoxon High vs Low (subject-level, REGIONAL) ---")
    L(paired.to_string(index=False))

    # Mixed-effects
    def fit_mlm(dv, data):
        try:
            md = smf.mixedlm(f"{dv} ~ C(phase, Treatment(reference='low_cognitive_load'))",
                             data, groups=data["subject"], re_formula="~1")
            return md.fit(method="lbfgs", reml=True)
        except Exception as e:
            return str(e)

    L("\n--- Mixed-Effects Model REGIONAL (log_ratio, window-level) ---")
    mlm1 = fit_mlm("log_ratio", W)
    mlm1_txt = mlm1.summary().as_text() if hasattr(mlm1, "summary") else str(mlm1)
    L(mlm1_txt)

    L("\n--- Mixed-Effects Model REGIONAL (log_ratio_norm, window-level) ---")
    Wn = W.dropna(subset=["log_ratio_norm"]).reset_index(drop=True)
    mlm2 = fit_mlm("log_ratio_norm", Wn)
    mlm2_txt = mlm2.summary().as_text() if hasattr(mlm2, "summary") else str(mlm2)
    L(mlm2_txt)

    with open(OUT / "_mixed_effects.txt", "w") as f:
        f.write("REGIONAL (F3/Fz/F4 theta ; P3/Pz/P4 alpha)\n")
        f.write("=" * 70 + "\n")
        f.write("log_ratio ~ phase + (1|subject)\n\n")
        f.write(mlm1_txt + "\n\n")
        f.write("log_ratio_norm ~ phase + (1|subject)\n\n")
        f.write(mlm2_txt + "\n")

    if hasattr(mlm1, "params"):
        plot_mlm_forest(mlm1, OUT / "_figure_mlm_forest.png")
        plot_mlm_forest(mlm1, THESIS_FIG)
        L(f"Forest plot MLM -> {THESIS_FIG}")

    # Figures
    order = ["baseline_eyes_open", "baseline_eyes_closed", "low_cognitive_load", "high_cognitive_load"]
    short = {"baseline_eyes_open": "base_open", "baseline_eyes_closed": "base_closed",
             "low_cognitive_load": "low", "high_cognitive_load": "high"}
    colors = ["#6baed6", "#2171b5", "#fdae6b", "#d94801"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    for ax, metric, title in [
        (axes[0], "mean_ratio", "θ/α regional (raw)"),
        (axes[1], "mean_logratio", "log(θ/α) regional"),
        (axes[2], "mean_logratio_norm", "log(θ/α) regional − base_open"),
    ]:
        piv = sm.pivot_table(index="subject", columns="phase", values=metric).reindex(columns=order)
        x = np.arange(len(order))
        means = piv.mean().values
        sems = piv.sem().values
        ax.bar(x, means, yerr=sems, color=colors, alpha=0.75, capsize=6, edgecolor="k", lw=0.6)
        for _, row in piv.iterrows():
            ax.plot(x, row.values, "o-", color="gray", lw=0.7, ms=3, alpha=0.55)
        ax.set_xticks(x); ax.set_xticklabels([short[p] for p in order], rotation=15)
        ax.set_title(f"{title}  (N={piv.dropna().shape[0]})")
        ax.grid(alpha=0.25)
        if metric == "mean_logratio_norm": ax.axhline(0, color="k", lw=0.8)
    fig.suptitle("REGIONAL — comparación de métricas (líneas grises = sujetos)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(OUT / "_figure_comparison.png", dpi=130)
    plt.close()

    fig, ax = plt.subplots(figsize=(10, 5.5))
    for i, ph in enumerate(order):
        vals = W["log_ratio_norm"][W["phase"] == ph].values
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0: continue
        parts = ax.violinplot([vals], positions=[i], widths=0.75, showmeans=True, showmedians=False)
        for pc in parts["bodies"]:
            pc.set_facecolor(colors[i]); pc.set_alpha(0.6); pc.set_edgecolor("k")
    ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.set_xticks(range(len(order))); ax.set_xticklabels([short[p] for p in order])
    ax.set_ylabel("log(θ/α) regional normalizado (− baseline_open)")
    ax.set_title("REGIONAL — distribución window-level log-ratio normalizado (pooled)")
    ax.grid(alpha=0.25)
    plt.tight_layout()
    fig.savefig(OUT / "_figure_window_violin.png", dpi=130)
    plt.close()

    with open(OUT / "_log.txt", "w") as f:
        f.write("\n".join(log_lines))

    L(f"\nSalidas en {OUT}")


if __name__ == "__main__":
    main()
