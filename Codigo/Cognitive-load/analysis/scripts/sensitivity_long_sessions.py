"""
Sensitivity analysis: only subjects with LONG experimental sessions.

Filter criterion (per session):
  - n_windows_after >= MIN_WINDOWS_LONG in BOTH low and high phases (and both baselines)
  - fz_artifact_pct and pz_artifact_pct <= 50 in both phases

MIN_WINDOWS_LONG = 10 cleanly separates the two groups in this dataset
(full sessions have 30-70 windows, truncated ones have 1-2).

This answers: "what do the stats look like if we drop all the
timer-truncated sessions and keep only the ones that ran to completion?"

Outputs (to output/statistical_validity/long_sessions/):
  - per_subject_data.csv
  - statistical_tests.csv
  - results_summary.csv
  - log.txt
  - figures/*.png
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple, Dict, List

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

# -------------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[2]
ANALYSIS_OUTPUT = BASE_DIR / "output" / "analysis_output"
OUT_DIR = BASE_DIR / "output" / "statistical_validity" / "long_sessions"
FIG_DIR = OUT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

CLEANED_FZPZ = ANALYSIS_OUTPUT / "cognitive_load_cleaned_summary_fz_pz.csv"
CLEANED_REGION = ANALYSIS_OUTPUT / "cognitive_load_cleaned_summary_region.csv"

LOG_PATH = OUT_DIR / "log.txt"

# -------------------------------------------------------------------------
# Inclusion criteria
# -------------------------------------------------------------------------
ARTIFACT_PCT_MAX = 50.0
MIN_WINDOWS_LONG = 10        # long-session threshold
PHASES = ["baseline_eyes_open", "baseline_eyes_closed",
          "low_cognitive_load", "high_cognitive_load"]


class Tee:
    def __init__(self, path: Path):
        self.f = open(path, "w", encoding="utf-8")

    def __call__(self, *args):
        msg = " ".join(str(a) for a in args)
        print(msg)
        self.f.write(msg + "\n"); self.f.flush()

    def close(self):
        self.f.close()


log = Tee(LOG_PATH)


def inclusion_long(df: pd.DataFrame) -> Tuple[List[str], Dict[str, str]]:
    """Long-session inclusion:
       - low + high both present
       - n_windows_after >= MIN_WINDOWS_LONG in BOTH low and high
       - artifact% <= 50 in both phases
    """
    included, excluded = [], {}
    for subj in df["subject"].unique():
        sub = df[df["subject"] == subj]
        low = sub[sub["phase"] == "low_cognitive_load"]
        high = sub[sub["phase"] == "high_cognitive_load"]
        if len(low) == 0 or len(high) == 0:
            excluded[subj] = "falta low o high"
            continue
        low = low.iloc[0]; high = high.iloc[0]
        reasons = []
        if low["n_windows_after"] < MIN_WINDOWS_LONG:
            reasons.append(f"low n_windows={int(low['n_windows_after'])}<{MIN_WINDOWS_LONG}")
        if high["n_windows_after"] < MIN_WINDOWS_LONG:
            reasons.append(f"high n_windows={int(high['n_windows_after'])}<{MIN_WINDOWS_LONG}")
        if low["fz_artifact_pct"] > ARTIFACT_PCT_MAX or low["pz_artifact_pct"] > ARTIFACT_PCT_MAX:
            reasons.append(f"low art Fz={low['fz_artifact_pct']:.0f}% Pz={low['pz_artifact_pct']:.0f}%")
        if high["fz_artifact_pct"] > ARTIFACT_PCT_MAX or high["pz_artifact_pct"] > ARTIFACT_PCT_MAX:
            reasons.append(f"high art Fz={high['fz_artifact_pct']:.0f}% Pz={high['pz_artifact_pct']:.0f}%")
        if reasons:
            excluded[subj] = "; ".join(reasons)
        else:
            included.append(subj)
    return included, excluded


def extract_paired(df, subjects, phase_a, phase_b):
    rows = []
    for s in subjects:
        a = df[(df["subject"] == s) & (df["phase"] == phase_a)]
        b = df[(df["subject"] == s) & (df["phase"] == phase_b)]
        if len(a) == 0 or len(b) == 0:
            continue
        va = float(a.iloc[0]["mean_ratio_after"])
        vb = float(b.iloc[0]["mean_ratio_after"])
        if not (np.isfinite(va) and np.isfinite(vb) and va > 0 and vb > 0):
            continue
        rows.append({"subject": s, "val_a": va, "val_b": vb})
    return pd.DataFrame(rows)


def cohens_dz(diffs):
    if len(diffs) < 2: return float("nan")
    s = np.std(diffs, ddof=1)
    return float(np.mean(diffs) / s) if s > 0 else float("nan")


def rank_biserial_paired(a, b):
    diffs = b - a
    diffs = diffs[diffs != 0]
    if len(diffs) == 0: return float("nan")
    ranks = stats.rankdata(np.abs(diffs))
    pos = ranks[diffs > 0].sum(); neg = ranks[diffs < 0].sum()
    tot = pos + neg
    return float((pos - neg) / tot) if tot > 0 else float("nan")


def bootstrap_ci(x, stat_fn, n_boot=10000, alpha=0.05, seed=42):
    if len(x) == 0: return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    n = len(x); boot = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        boot[i] = stat_fn(x[idx])
    return float(np.percentile(boot, 100*alpha/2)), float(np.percentile(boot, 100*(1-alpha/2)))


def wilcoxon_safe(a, b, alt="two-sided"):
    diffs = b - a
    if np.all(diffs == 0): return float("nan"), float("nan")
    try:
        r = stats.wilcoxon(a, b, alternative=alt, zero_method="wilcox")
        return float(r.statistic), float(r.pvalue)
    except ValueError:
        return float("nan"), float("nan")


def paired_t_on_log(a, b, alt="greater"):
    mask = (a > 0) & (b > 0)
    la = np.log(a[mask]); lb = np.log(b[mask])
    if len(la) < 2: return float("nan"), float("nan")
    r = stats.ttest_rel(lb, la, alternative=alt)
    return float(r.statistic), float(r.pvalue)


def sign_test(a, b, alt="greater"):
    diffs = b - a
    n_pos = int(np.sum(diffs > 0)); n_neg = int(np.sum(diffs < 0))
    n = n_pos + n_neg
    if n == 0: return 0, float("nan")
    if alt == "greater":
        p = stats.binomtest(n_pos, n, p=0.5, alternative="greater").pvalue
    elif alt == "less":
        p = stats.binomtest(n_pos, n, p=0.5, alternative="less").pvalue
    else:
        p = stats.binomtest(n_pos, n, p=0.5, alternative="two-sided").pvalue
    return n_pos, float(p)


def run_contrast(df, subjects, label, phase_a, phase_b, alternative="greater"):
    paired = extract_paired(df, subjects, phase_a, phase_b)
    if len(paired) < 2:
        return None
    a = paired["val_a"].to_numpy(); b = paired["val_b"].to_numpy()
    diffs = b - a
    W, p_w = wilcoxon_safe(a, b, alternative)
    t, p_t = paired_t_on_log(a, b, alternative)
    dz = cohens_dz(diffs)
    rb = rank_biserial_paired(a, b)
    ci_lo, ci_hi = bootstrap_ci(diffs, np.mean)
    n_pos, p_sign = sign_test(a, b, alternative)
    return {
        "contrast": label,
        "phase_a": phase_a, "phase_b": phase_b,
        "n": int(len(paired)),
        "mean_a": float(np.mean(a)), "mean_b": float(np.mean(b)),
        "mean_diff": float(np.mean(diffs)),
        "median_diff": float(np.median(diffs)),
        "wilcoxon_W": W, "wilcoxon_p": p_w,
        "ttest_log_t": t, "ttest_log_p": p_t,
        "cohens_dz": dz, "rank_biserial_r": rb,
        "n_b_greater_a": n_pos, "sign_p": p_sign,
        "ci_low_mean_diff": ci_lo, "ci_high_mean_diff": ci_hi,
        "alternative": alternative,
    }


CONTRASTS = [
    ("H1_High_vs_Low",          "low_cognitive_load",   "high_cognitive_load",  "greater"),
    ("H2a_High_vs_BaseOpen",    "baseline_eyes_open",   "high_cognitive_load",  "greater"),
    ("H2b_High_vs_BaseClosed",  "baseline_eyes_closed", "high_cognitive_load",  "greater"),
    ("H3a_Low_vs_BaseOpen",     "baseline_eyes_open",   "low_cognitive_load",   "two-sided"),
    ("H3b_Low_vs_BaseClosed",   "baseline_eyes_closed", "low_cognitive_load",   "two-sided"),
    ("H4_BaseOpen_vs_BaseClosed","baseline_eyes_open",  "baseline_eyes_closed", "two-sided"),
]


def build_per_subject_rows(df, subjects, dataset_label):
    rows = []
    for s in subjects:
        row = {"subject": s, "dataset": dataset_label}
        for ph in PHASES:
            r = df[(df["subject"] == s) & (df["phase"] == ph)]
            if len(r):
                r = r.iloc[0]
                row[f"{ph}_mean"] = float(r["mean_ratio_after"])
                row[f"{ph}_n_windows"] = int(r["n_windows_after"])
                row[f"{ph}_fz_art_pct"] = float(r.get("fz_artifact_pct", np.nan))
                row[f"{ph}_pz_art_pct"] = float(r.get("pz_artifact_pct", np.nan))
        rows.append(row)
    return rows


def plot_paired_slope_log(df, subjects, title, out_path):
    data = extract_paired(df, subjects, "low_cognitive_load", "high_cognitive_load")
    if len(data) < 2: return
    fig, ax = plt.subplots(figsize=(6, 5))
    for _, r in data.iterrows():
        ax.plot([0, 1], [r["val_a"], r["val_b"]], "-o", color="#4c72b0", alpha=0.6)
    gm_low = np.exp(np.mean(np.log(data["val_a"])))
    gm_high = np.exp(np.mean(np.log(data["val_b"])))
    ax.plot([0, 1], [gm_low, gm_high], "-s", color="#c44e52", lw=3, ms=12, label="Geometric mean")
    ax.set_yscale("log")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Low Load", "High Load"], fontsize=12)
    ax.set_ylabel("Cognitive Load Ratio (Theta/Alpha) [log]")
    ax.set_title(title)
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3, which="both")
    plt.tight_layout(); plt.savefig(out_path, dpi=150); plt.close()


def plot_delta_forest(df, subjects, title, out_path):
    data = extract_paired(df, subjects, "low_cognitive_load", "high_cognitive_load")
    if len(data) < 2: return
    data["delta"] = data["val_b"] - data["val_a"]
    data = data.sort_values("delta", ascending=False)
    fig, ax = plt.subplots(figsize=(7, max(4, 0.5 * len(data))))
    colors = ["#55a868" if d > 0 else "#c44e52" for d in data["delta"]]
    ax.barh(data["subject"], data["delta"], color=colors, edgecolor="k")
    mean_delta = float(np.mean(data["delta"]))
    ci_lo, ci_hi = bootstrap_ci(data["delta"].to_numpy(), np.mean)
    ax.axvline(mean_delta, color="#4c72b0", ls="--", lw=2,
               label=f"Mean = {mean_delta:.2f}  [95% CI: {ci_lo:.2f}, {ci_hi:.2f}]")
    ax.axvline(0, color="k", lw=1)
    ax.set_xlabel("High − Low Ratio")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout(); plt.savefig(out_path, dpi=150); plt.close()


def plot_boxplot(df, subjects, title, out_path):
    vals = {ph: [] for ph in PHASES}
    for s in subjects:
        for ph in PHASES:
            r = df[(df["subject"] == s) & (df["phase"] == ph)]
            if len(r): vals[ph].append(float(r.iloc[0]["mean_ratio_after"]))
    fig, ax = plt.subplots(figsize=(7, 5))
    box = ax.boxplot([vals[ph] for ph in PHASES],
                     labels=["BaseOpen", "BaseClosed", "LowLoad", "HighLoad"],
                     patch_artist=True, showmeans=True)
    for p, c in zip(box["boxes"], ["#6baed6", "#2171b5", "#fdae6b", "#d94801"]):
        p.set_facecolor(c); p.set_alpha(0.7)
    ax.set_yscale("log")
    ax.set_ylabel("Cognitive Load Ratio (Theta/Alpha) [log]")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3, which="both")
    plt.tight_layout(); plt.savefig(out_path, dpi=150); plt.close()


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------
def main():
    log("=" * 72)
    log("SENSITIVITY ANALYSIS — LONG SESSIONS ONLY")
    log(f"Criterion: n_windows_after >= {MIN_WINDOWS_LONG} AND artifact_pct <= {ARTIFACT_PCT_MAX:.0f}% in low & high")
    log("=" * 72)

    df_fz = pd.read_csv(CLEANED_FZPZ)
    df_rg = pd.read_csv(CLEANED_REGION)
    log(f"Loaded Fz/Pz: {len(df_fz)} rows, {df_fz['subject'].nunique()} subjects")
    log(f"Loaded Region: {len(df_rg)} rows, {df_rg['subject'].nunique()} subjects")

    inc_fz, exc_fz = inclusion_long(df_fz)
    inc_rg, exc_rg = inclusion_long(df_rg)

    log("\nLong-session Fz/Pz (included):", len(inc_fz))
    log("  " + ", ".join(inc_fz))
    log("Long-session Fz/Pz (excluded):", len(exc_fz))
    for s, why in exc_fz.items():
        log(f"  - {s}: {why}")

    log("\nLong-session Region (included):", len(inc_rg))
    log("  " + ", ".join(inc_rg))
    log("Long-session Region (excluded):", len(exc_rg))
    for s, why in exc_rg.items():
        log(f"  - {s}: {why}")

    # per-subject data
    rows = build_per_subject_rows(df_fz, inc_fz, "fz_pz") + \
           build_per_subject_rows(df_rg, inc_rg, "region")
    pd.DataFrame(rows).to_csv(OUT_DIR / "per_subject_data.csv", index=False)
    log("\nSaved per_subject_data.csv")

    # contrasts
    all_results = []
    for label, pa, pb, alt in CONTRASTS:
        r = run_contrast(df_fz, inc_fz, label, pa, pb, alt)
        if r: r["dataset"] = "fz_pz"; all_results.append(r)
    for label, pa, pb, alt in CONTRASTS:
        r = run_contrast(df_rg, inc_rg, label, pa, pb, alt)
        if r: r["dataset"] = "region"; all_results.append(r)

    df_res = pd.DataFrame(all_results)
    cols = ["dataset", "contrast", "phase_a", "phase_b", "n",
            "mean_a", "mean_b", "mean_diff", "median_diff",
            "wilcoxon_W", "wilcoxon_p", "ttest_log_t", "ttest_log_p",
            "cohens_dz", "rank_biserial_r",
            "n_b_greater_a", "sign_p",
            "ci_low_mean_diff", "ci_high_mean_diff", "alternative"]
    df_res[cols].to_csv(OUT_DIR / "statistical_tests.csv", index=False)

    # compact summary
    summary_rows = []
    for r in all_results:
        summary_rows.append({
            "dataset": r["dataset"],
            "contrast": r["contrast"],
            "n": r["n"],
            "mean_phase_a": round(r["mean_a"], 4),
            "mean_phase_b": round(r["mean_b"], 4),
            "mean_diff_b_minus_a": round(r["mean_diff"], 4),
            "ci95_mean_diff": f"[{r['ci_low_mean_diff']:.3f}, {r['ci_high_mean_diff']:.3f}]",
            "wilcoxon_p": round(r["wilcoxon_p"], 3),
            "paired_t_log_p": round(r["ttest_log_p"], 3),
            "cohens_dz": round(r["cohens_dz"], 3),
            "rank_biserial_r": round(r["rank_biserial_r"], 3),
            "signs_b_gt_a": f"{r['n_b_greater_a']}/{r['n']}",
            "sign_test_p": round(r["sign_p"], 3),
            "alternative": r["alternative"],
        })
    pd.DataFrame(summary_rows).to_csv(OUT_DIR / "results_summary.csv", index=False)

    # print tables
    for ds in ["fz_pz", "region"]:
        log(f"\n--- {ds} ---")
        for r in all_results:
            if r["dataset"] != ds: continue
            log(f"{r['contrast']:<28} n={r['n']:<3}  mean_a={r['mean_a']:.3f}  mean_b={r['mean_b']:.3f}  "
                f"Δ={r['mean_diff']:+.3f}  [CI {r['ci_low_mean_diff']:+.2f}, {r['ci_high_mean_diff']:+.2f}]  "
                f"Wilcoxon p={r['wilcoxon_p']:.3f}  t(log) p={r['ttest_log_p']:.3f}  dz={r['cohens_dz']:+.2f}  "
                f"signs={r['n_b_greater_a']}/{r['n']} (p={r['sign_p']:.3f})")

    # figures
    log("\nFigures")
    if inc_fz:
        plot_paired_slope_log(df_fz, inc_fz, "Paired change Low → High  (Fz/Pz, long sessions)",
                              FIG_DIR / "01_paired_slope_fzpz_log.png")
        plot_delta_forest(df_fz, inc_fz, "Individual deltas (High − Low) — Fz/Pz (long sessions)",
                          FIG_DIR / "02_delta_forest_fzpz.png")
        plot_boxplot(df_fz, inc_fz, "Ratio by phase — Fz/Pz (long sessions)",
                     FIG_DIR / "03_boxplot_fzpz.png")
        log("  Saved Fz/Pz figures")
    if inc_rg:
        plot_paired_slope_log(df_rg, inc_rg, "Paired change Low → High  (regions, long sessions)",
                              FIG_DIR / "04_paired_slope_region_log.png")
        plot_delta_forest(df_rg, inc_rg, "Individual deltas (High − Low) — regions (long sessions)",
                          FIG_DIR / "05_delta_forest_region.png")
        plot_boxplot(df_rg, inc_rg, "Ratio by phase — regions (long sessions)",
                     FIG_DIR / "06_boxplot_region.png")
        log("  Saved Region figures")

    log("\nDONE.")
    log.close()


if __name__ == "__main__":
    main()
