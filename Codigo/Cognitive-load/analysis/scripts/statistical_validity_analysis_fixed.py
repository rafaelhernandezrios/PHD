"""
Statistical validity analysis of the Theta/Alpha cognitive load index.

Input:
  - output/analysis_output/cognitive_load_cleaned_summary_fz_pz.csv  (Fz/Pz, primary)
  - output/analysis_output/cognitive_load_cleaned_summary_region.csv (F/P regions, sensitivity)
  - nasa_tlx.xlsx (exploratory convergent validity)

Inclusion criteria (per session):
  - low_cognitive_load and high_cognitive_load present
  - fz_artifact_pct <= 50 and pz_artifact_pct <= 50 in BOTH low and high phases
  - n_windows_after >= 2 in BOTH low and high phases

Tests:
  H1 (primary):   High Load > Low Load          -> Wilcoxon paired + t-test on log(ratio) paired
  H2 (secondary): High Load > Baseline          -> Wilcoxon paired
  H3 (secondary): Low  Load vs Baseline         -> Wilcoxon paired (no directional prediction)
  H4 (exploratory): Spearman NASA-TLX vs EEG    -> N ~ 5

Effect sizes: Cohen's dz (paired), rank-biserial r (Wilcoxon), 95% CIs (bootstrap 10,000 iters).

Outputs (to output/statistical_validity/):
  - per_subject_data.csv
  - results_summary.csv
  - statistical_tests.csv
  - nasa_tlx_merged.csv
  - figures/*.png and *.pdf
  - log.txt

Author: Rafael — April 2026
"""

from __future__ import annotations

import os
import sys
import json
from pathlib import Path
from typing import Tuple, Dict, List

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

# -------------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[2]                       # Cognitive-load/
ANALYSIS_OUTPUT = BASE_DIR / "output" / "analysis_output_fixed"
OUT_DIR = BASE_DIR / "output" / "statistical_validity_fixed"
FIG_DIR = OUT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

CLEANED_FZPZ = ANALYSIS_OUTPUT / "cognitive_load_cleaned_summary_fz_pz.csv"
CLEANED_REGION = ANALYSIS_OUTPUT / "cognitive_load_cleaned_summary_region.csv"
NASA_TLX_XLSX = BASE_DIR / "nasa_tlx.xlsx"

LOG_PATH = OUT_DIR / "log.txt"

# -------------------------------------------------------------------------
# Inclusion criteria
# -------------------------------------------------------------------------
ARTIFACT_PCT_MAX = 50.0
MIN_WINDOWS = 2

PHASES = ["baseline_eyes_open", "baseline_eyes_closed",
          "low_cognitive_load", "high_cognitive_load"]


# -------------------------------------------------------------------------
# Logging
# -------------------------------------------------------------------------
class Tee:
    """Write to stdout AND log file at the same time."""

    def __init__(self, path: Path):
        self.path = path
        self.f = open(path, "w", encoding="utf-8")

    def __call__(self, *args, **kw):
        msg = " ".join(str(a) for a in args)
        print(msg, **kw)
        self.f.write(msg + "\n")
        self.f.flush()

    def close(self):
        self.f.close()


log = Tee(LOG_PATH)


# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------
def apply_inclusion(df: pd.DataFrame) -> Tuple[List[str], Dict[str, str]]:
    """Apply inclusion criteria to a summary dataframe. Return (included, excluded_with_reasons)."""
    included = []
    excluded: Dict[str, str] = {}
    for subj in df["subject"].unique():
        sub = df[df["subject"] == subj]
        low = sub[sub["phase"] == "low_cognitive_load"]
        high = sub[sub["phase"] == "high_cognitive_load"]
        if len(low) == 0 or len(high) == 0:
            excluded[subj] = "falta low o high"
            continue
        low = low.iloc[0]
        high = high.iloc[0]
        reasons = []
        if low["fz_artifact_pct"] > ARTIFACT_PCT_MAX or low["pz_artifact_pct"] > ARTIFACT_PCT_MAX:
            reasons.append(f"low art Fz={low['fz_artifact_pct']:.0f}% Pz={low['pz_artifact_pct']:.0f}%")
        if high["fz_artifact_pct"] > ARTIFACT_PCT_MAX or high["pz_artifact_pct"] > ARTIFACT_PCT_MAX:
            reasons.append(f"high art Fz={high['fz_artifact_pct']:.0f}% Pz={high['pz_artifact_pct']:.0f}%")
        if low["n_windows_after"] < MIN_WINDOWS:
            reasons.append(f"low n_windows={int(low['n_windows_after'])}")
        if high["n_windows_after"] < MIN_WINDOWS:
            reasons.append(f"high n_windows={int(high['n_windows_after'])}")
        if reasons:
            excluded[subj] = "; ".join(reasons)
        else:
            included.append(subj)
    return included, excluded


def extract_paired(df: pd.DataFrame, subjects: List[str],
                   phase_a: str, phase_b: str,
                   value_col: str = "mean_ratio_after") -> pd.DataFrame:
    """Return a dataframe with subject, val_a, val_b columns for a paired test."""
    rows = []
    for s in subjects:
        a = df[(df["subject"] == s) & (df["phase"] == phase_a)]
        b = df[(df["subject"] == s) & (df["phase"] == phase_b)]
        if len(a) == 0 or len(b) == 0:
            continue
        va = float(a.iloc[0][value_col])
        vb = float(b.iloc[0][value_col])
        if not (np.isfinite(va) and np.isfinite(vb) and va > 0 and vb > 0):
            continue
        rows.append({"subject": s, "val_a": va, "val_b": vb})
    return pd.DataFrame(rows)


def cohens_dz(diffs: np.ndarray) -> float:
    """Cohen's dz for paired design: mean(diff) / sd(diff)."""
    if len(diffs) < 2:
        return float("nan")
    s = np.std(diffs, ddof=1)
    return float(np.mean(diffs) / s) if s > 0 else float("nan")


def rank_biserial_paired(a: np.ndarray, b: np.ndarray) -> float:
    """Matched-pairs rank-biserial correlation (Kerby 2014).
    r = (W+ - W-) / (W+ + W-)  where W are rank sums of positive/negative diffs.
    Bounded in [-1, 1]. Positive means b > a."""
    diffs = b - a
    diffs = diffs[diffs != 0]
    if len(diffs) == 0:
        return float("nan")
    ranks = stats.rankdata(np.abs(diffs))
    pos = ranks[diffs > 0].sum()
    neg = ranks[diffs < 0].sum()
    tot = pos + neg
    return float((pos - neg) / tot) if tot > 0 else float("nan")


def bootstrap_ci(x: np.ndarray, stat_fn, n_boot: int = 10000,
                 alpha: float = 0.05, rng_seed: int = 42) -> Tuple[float, float]:
    """Percentile bootstrap CI for a statistic over a 1D array."""
    if len(x) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(rng_seed)
    boot = np.empty(n_boot)
    n = len(x)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        boot[i] = stat_fn(x[idx])
    lo = float(np.percentile(boot, 100 * alpha / 2))
    hi = float(np.percentile(boot, 100 * (1 - alpha / 2)))
    return lo, hi


def wilcoxon_safe(a: np.ndarray, b: np.ndarray, alternative: str = "two-sided"):
    """Wilcoxon paired; catches the 'zero diffs' edge case."""
    diffs = b - a
    if np.all(diffs == 0):
        return float("nan"), float("nan")
    try:
        res = stats.wilcoxon(a, b, alternative=alternative, zero_method="wilcox")
        return float(res.statistic), float(res.pvalue)
    except ValueError:
        return float("nan"), float("nan")


def paired_t_on_log(a: np.ndarray, b: np.ndarray, alternative: str = "greater"):
    """Paired t-test on log-transformed ratios (handles skewness)."""
    la = np.log(a)
    lb = np.log(b)
    res = stats.ttest_rel(lb, la, alternative=alternative)
    return float(res.statistic), float(res.pvalue)


def fmt_p(p: float) -> str:
    if not np.isfinite(p):
        return "NA"
    if p < 0.001:
        return "<0.001"
    return f"{p:.3f}"


# -------------------------------------------------------------------------
# Main tests
# -------------------------------------------------------------------------
def run_primary_tests(df: pd.DataFrame, label: str, subjects: List[str]) -> List[Dict]:
    """Run H1–H3 tests on the given summary dataframe."""
    results = []

    contrasts = [
        ("H1_High_vs_Low", "low_cognitive_load", "high_cognitive_load", "greater"),
        ("H2a_High_vs_BaseOpen", "baseline_eyes_open", "high_cognitive_load", "greater"),
        ("H2b_High_vs_BaseClosed", "baseline_eyes_closed", "high_cognitive_load", "greater"),
        ("H3a_Low_vs_BaseOpen", "baseline_eyes_open", "low_cognitive_load", "two-sided"),
        ("H3b_Low_vs_BaseClosed", "baseline_eyes_closed", "low_cognitive_load", "two-sided"),
        ("H4_BaseOpen_vs_BaseClosed", "baseline_eyes_open", "baseline_eyes_closed", "two-sided"),
    ]

    for name, pa, pb, alt in contrasts:
        paired = extract_paired(df, subjects, pa, pb)
        if len(paired) < 2:
            results.append({
                "dataset": label, "contrast": name, "phase_a": pa, "phase_b": pb,
                "n": len(paired), "mean_a": np.nan, "mean_b": np.nan,
                "mean_diff": np.nan, "median_diff": np.nan,
                "wilcoxon_W": np.nan, "wilcoxon_p": np.nan,
                "ttest_log_t": np.nan, "ttest_log_p": np.nan,
                "cohens_dz": np.nan, "rank_biserial_r": np.nan,
                "n_b_greater_a": 0, "sign_p": np.nan,
                "ci_low_mean_diff": np.nan, "ci_high_mean_diff": np.nan,
                "alternative": alt,
            })
            continue

        a = paired["val_a"].values
        b = paired["val_b"].values
        diffs = b - a

        w_stat, w_p = wilcoxon_safe(a, b, alternative=alt)
        t_stat, t_p = paired_t_on_log(a, b, alternative=alt)
        dz = cohens_dz(diffs)
        rb = rank_biserial_paired(a, b)
        ci_lo, ci_hi = bootstrap_ci(diffs, np.mean)

        n_b_gt_a = int(np.sum(b > a))
        n = len(paired)
        # Binomial sign test (two-sided OR directional per alt)
        if alt == "greater":
            sign_p = float(stats.binomtest(n_b_gt_a, n, 0.5, alternative="greater").pvalue)
        else:
            sign_p = float(stats.binomtest(n_b_gt_a, n, 0.5, alternative="two-sided").pvalue)

        results.append({
            "dataset": label, "contrast": name, "phase_a": pa, "phase_b": pb,
            "n": n, "mean_a": float(np.mean(a)), "mean_b": float(np.mean(b)),
            "mean_diff": float(np.mean(diffs)), "median_diff": float(np.median(diffs)),
            "wilcoxon_W": w_stat, "wilcoxon_p": w_p,
            "ttest_log_t": t_stat, "ttest_log_p": t_p,
            "cohens_dz": dz, "rank_biserial_r": rb,
            "n_b_greater_a": n_b_gt_a, "sign_p": sign_p,
            "ci_low_mean_diff": ci_lo, "ci_high_mean_diff": ci_hi,
            "alternative": alt,
        })

    return results


# -------------------------------------------------------------------------
# NASA-TLX ingestion
# -------------------------------------------------------------------------
def load_nasa_tlx() -> pd.DataFrame:
    """Return dataframe with columns: person, nasa_tlx_pct (Stroop/Carga cognitiva)."""
    if not NASA_TLX_XLSX.is_file():
        return pd.DataFrame(columns=["person", "nasa_tlx_pct"])
    try:
        df = pd.read_excel(NASA_TLX_XLSX, sheet_name="Resultados")
    except Exception:
        return pd.DataFrame(columns=["person", "nasa_tlx_pct"])
    df = df.rename(columns={"Persona": "person", "% Carga cognitiva": "nasa_tlx_pct"})
    return df[["person", "nasa_tlx_pct"]].dropna()


# Map NASA-TLX person label -> EEG subject id (as appears in CSVs)
NASA_MAP = {
    "Joselin": "Joss",
    "Daniel": "Daniel",
    "Edwin": "Edwin",
    "Jeronimo": "Jeronimo",
    "Rafa": "Rafael",
    "Eliza": "eliza",
}


def nasa_merge(df_summary: pd.DataFrame, tlx: pd.DataFrame, included: List[str]) -> pd.DataFrame:
    """Merge NASA-TLX % with EEG subject-level ratios (for low/high)."""
    rows = []
    for _, r in tlx.iterrows():
        person = r["person"]
        if person not in NASA_MAP:
            continue
        subj = NASA_MAP[person]
        if subj not in included:
            continue
        low = df_summary[(df_summary["subject"] == subj) & (df_summary["phase"] == "low_cognitive_load")]
        high = df_summary[(df_summary["subject"] == subj) & (df_summary["phase"] == "high_cognitive_load")]
        if len(low) == 0 or len(high) == 0:
            continue
        rows.append({
            "tlx_person": person, "subject": subj,
            "nasa_tlx_pct": float(r["nasa_tlx_pct"]),
            "ratio_low": float(low.iloc[0]["mean_ratio_after"]),
            "ratio_high": float(high.iloc[0]["mean_ratio_after"]),
            "ratio_delta": float(high.iloc[0]["mean_ratio_after"]) - float(low.iloc[0]["mean_ratio_after"]),
        })
    return pd.DataFrame(rows)


# -------------------------------------------------------------------------
# Plots
# -------------------------------------------------------------------------
def set_paper_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 150,
    })


def save_fig(fig, name: str):
    for ext in ("png", "pdf"):
        path = FIG_DIR / f"{name}.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_paired_slope(paired: pd.DataFrame, title: str, ylabel: str, name: str):
    fig, ax = plt.subplots(figsize=(6, 5))
    xs = np.array([0, 1])
    for _, r in paired.iterrows():
        ax.plot(xs, [r["val_a"], r["val_b"]],
                marker="o", color="#2b5ea8", alpha=0.6, linewidth=1.5, markersize=6)
    # Group means
    ax.plot(xs, [paired["val_a"].mean(), paired["val_b"].mean()],
            marker="s", color="#c0392b", linewidth=3, markersize=10, label="Mean")
    ax.set_xticks(xs)
    ax.set_xticklabels(["Low Load", "High Load"])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.25, axis="y")
    ax.legend(loc="best", frameon=False)
    save_fig(fig, name)


def plot_paired_slope_log(paired: pd.DataFrame, title: str, name: str):
    fig, ax = plt.subplots(figsize=(6, 5))
    xs = np.array([0, 1])
    for _, r in paired.iterrows():
        ax.plot(xs, [r["val_a"], r["val_b"]],
                marker="o", color="#2b5ea8", alpha=0.6, linewidth=1.5, markersize=6)
    ax.plot(xs, [np.exp(np.mean(np.log(paired["val_a"]))),
                 np.exp(np.mean(np.log(paired["val_b"])))],
            marker="s", color="#c0392b", linewidth=3, markersize=10,
            label="Geometric mean")
    ax.set_xticks(xs)
    ax.set_xticklabels(["Low Load", "High Load"])
    ax.set_yscale("log")
    ax.set_ylabel("Cognitive Load Ratio (Theta/Alpha)  [log scale]")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.25, axis="y")
    ax.legend(loc="best", frameon=False)
    save_fig(fig, name)


def plot_delta_forest(paired: pd.DataFrame, title: str, name: str):
    """Per-subject delta with bootstrap CI."""
    fig, ax = plt.subplots(figsize=(6, max(3.5, 0.35 * len(paired) + 2)))
    paired_sorted = paired.copy()
    paired_sorted["delta"] = paired_sorted["val_b"] - paired_sorted["val_a"]
    paired_sorted = paired_sorted.sort_values("delta")
    y = np.arange(len(paired_sorted))
    colors = ["#27ae60" if d > 0 else "#c0392b" for d in paired_sorted["delta"]]
    ax.barh(y, paired_sorted["delta"], color=colors, alpha=0.8, edgecolor="black")
    ax.axvline(0, color="black", linewidth=1)
    # Mean + CI
    mean_d = paired_sorted["delta"].mean()
    ci_lo, ci_hi = bootstrap_ci(paired_sorted["delta"].values, np.mean)
    ax.axvline(mean_d, color="#2b5ea8", linewidth=2, linestyle="--",
               label=f"Mean = {mean_d:.2f}  [95% CI: {ci_lo:.2f}, {ci_hi:.2f}]")
    ax.set_yticks(y)
    ax.set_yticklabels(paired_sorted["subject"])
    ax.set_xlabel("High − Low Ratio")
    ax.set_title(title)
    ax.grid(True, alpha=0.25, axis="x")
    ax.legend(loc="best", frameon=False, fontsize=9)
    save_fig(fig, name)


def plot_boxplot(df: pd.DataFrame, subjects: List[str], title: str, name: str):
    """Boxplot of Baseline / Low / High for included subjects."""
    data = []
    labels = []
    mapping = [
        ("baseline_eyes_open", "Baseline\nopen"),
        ("baseline_eyes_closed", "Baseline\nclosed"),
        ("low_cognitive_load", "Low Load"),
        ("high_cognitive_load", "High Load"),
    ]
    for phase, lab in mapping:
        vals = []
        for s in subjects:
            r = df[(df["subject"] == s) & (df["phase"] == phase)]
            if len(r) > 0:
                v = float(r.iloc[0]["mean_ratio_after"])
                if np.isfinite(v) and v > 0:
                    vals.append(v)
        data.append(vals)
        labels.append(lab)

    fig, ax = plt.subplots(figsize=(7, 5))
    bp = ax.boxplot(data, tick_labels=labels, showfliers=False, patch_artist=True,
                    medianprops=dict(color="black", linewidth=2))
    palette = ["#4fc3f7", "#81c784", "#ffb74d", "#e57373"]
    for patch, c in zip(bp["boxes"], palette):
        patch.set_facecolor(c)
        patch.set_alpha(0.8)
    # Overlay subject points
    for i, vals in enumerate(data):
        xs = np.random.default_rng(i).normal(i + 1, 0.05, len(vals))
        ax.scatter(xs, vals, color="black", s=18, alpha=0.5, zorder=3)
    ax.set_ylabel("Cognitive Load Ratio (Theta/Alpha)")
    ax.set_title(title)
    ax.grid(True, alpha=0.25, axis="y")
    save_fig(fig, name)


def plot_nasa_scatter(merged: pd.DataFrame, y_col: str, ylabel: str, title: str, name: str):
    if len(merged) < 3:
        return
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(merged["nasa_tlx_pct"], merged[y_col], s=70, color="#2b5ea8", alpha=0.8, edgecolor="black")
    for _, r in merged.iterrows():
        ax.annotate(r["subject"], (r["nasa_tlx_pct"], r[y_col]),
                    textcoords="offset points", xytext=(6, 4), fontsize=9)
    rho, p = stats.spearmanr(merged["nasa_tlx_pct"], merged[y_col])
    ax.set_xlabel("NASA-TLX (%) — Stroop")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title}\nSpearman ρ = {rho:.2f}, p = {fmt_p(float(p))} (N={len(merged)})")
    ax.grid(True, alpha=0.25)
    save_fig(fig, name)


# -------------------------------------------------------------------------
# Orchestration
# -------------------------------------------------------------------------
def main():
    set_paper_style()
    log("=" * 72)
    log("STATISTICAL VALIDITY ANALYSIS — Theta/Alpha Cognitive Load Index")
    log("=" * 72)

    # 1) Load summaries
    if not CLEANED_FZPZ.is_file():
        log(f"ERROR: missing {CLEANED_FZPZ}. Run step4 first.")
        sys.exit(1)
    df_fzpz = pd.read_csv(CLEANED_FZPZ)
    log(f"Loaded Fz/Pz summary: {len(df_fzpz)} rows, subjects={df_fzpz['subject'].nunique()}")

    df_region = None
    if CLEANED_REGION.is_file():
        df_region = pd.read_csv(CLEANED_REGION)
        log(f"Loaded F/P region summary: {len(df_region)} rows")
    else:
        log(f"WARNING: no region summary at {CLEANED_REGION} — sensitivity analysis skipped.")

    # 2) Inclusion criteria
    incl_fzpz, excl_fzpz = apply_inclusion(df_fzpz)
    log(f"\nInclusion Fz/Pz — included (N={len(incl_fzpz)}):")
    log("  " + ", ".join(incl_fzpz))
    log(f"Excluded (N={len(excl_fzpz)}):")
    for s, r in sorted(excl_fzpz.items()):
        log(f"  - {s}: {r}")

    if df_region is not None:
        incl_region, excl_region = apply_inclusion(df_region)
        log(f"\nInclusion F/P regions — included (N={len(incl_region)}):")
        log("  " + ", ".join(incl_region))
        log(f"Excluded (N={len(excl_region)}):")
        for s, r in sorted(excl_region.items()):
            log(f"  - {s}: {r}")

    # 3) Per-subject tidy data
    per_subject_rows = []
    for s in incl_fzpz:
        sub = df_fzpz[df_fzpz["subject"] == s]
        row: Dict = {"subject": s, "dataset": "fz_pz"}
        for phase in PHASES:
            r = sub[sub["phase"] == phase]
            if len(r) > 0:
                row[f"{phase}_mean"] = float(r.iloc[0]["mean_ratio_after"])
                row[f"{phase}_median"] = float(r.iloc[0]["median_ratio_after"])
                row[f"{phase}_n_windows"] = int(r.iloc[0]["n_windows_after"])
                row[f"{phase}_fz_art_pct"] = float(r.iloc[0]["fz_artifact_pct"])
                row[f"{phase}_pz_art_pct"] = float(r.iloc[0]["pz_artifact_pct"])
        per_subject_rows.append(row)

    if df_region is not None:
        for s in incl_region:
            sub = df_region[df_region["subject"] == s]
            row = {"subject": s, "dataset": "region"}
            for phase in PHASES:
                r = sub[sub["phase"] == phase]
                if len(r) > 0:
                    row[f"{phase}_mean"] = float(r.iloc[0]["mean_ratio_after"])
                    row[f"{phase}_median"] = float(r.iloc[0]["median_ratio_after"])
                    row[f"{phase}_n_windows"] = int(r.iloc[0]["n_windows_after"])
                    row[f"{phase}_fz_art_pct"] = float(r.iloc[0]["fz_artifact_pct"])
                    row[f"{phase}_pz_art_pct"] = float(r.iloc[0]["pz_artifact_pct"])
            per_subject_rows.append(row)

    df_per = pd.DataFrame(per_subject_rows)
    df_per.to_csv(OUT_DIR / "per_subject_data.csv", index=False)
    log(f"\nSaved: {OUT_DIR / 'per_subject_data.csv'}")

    # 4) Tests
    log("\n" + "=" * 72)
    log("STATISTICAL TESTS")
    log("=" * 72)

    all_tests: List[Dict] = []
    all_tests.extend(run_primary_tests(df_fzpz, "fz_pz", incl_fzpz))
    if df_region is not None:
        all_tests.extend(run_primary_tests(df_region, "region", incl_region))

    df_tests = pd.DataFrame(all_tests)
    df_tests.to_csv(OUT_DIR / "statistical_tests.csv", index=False)
    log(f"\nSaved: {OUT_DIR / 'statistical_tests.csv'}")

    # Print compact results table
    log("")
    for ds in df_tests["dataset"].unique():
        log(f"\n--- {ds} ---")
        dsd = df_tests[df_tests["dataset"] == ds]
        for _, row in dsd.iterrows():
            log(f"{row['contrast']:<28} n={int(row['n']):<3}  "
                f"mean_a={row['mean_a']:<7.3f} mean_b={row['mean_b']:<7.3f}  "
                f"Δ={row['mean_diff']:<7.3f} [95%CI {row['ci_low_mean_diff']:+.2f}, {row['ci_high_mean_diff']:+.2f}]  "
                f"Wilcoxon p={fmt_p(row['wilcoxon_p']):<8} "
                f"t(log) p={fmt_p(row['ttest_log_p']):<8} "
                f"dz={row['cohens_dz']:+.2f}  rb={row['rank_biserial_r']:+.2f}  "
                f"signs={int(row['n_b_greater_a'])}/{int(row['n'])} (p={fmt_p(row['sign_p'])})")

    # 5) NASA-TLX — exploratory
    log("\n" + "=" * 72)
    log("NASA-TLX (exploratory convergent validity)")
    log("=" * 72)
    tlx = load_nasa_tlx()
    log(f"NASA-TLX sheet 'Resultados' — {len(tlx)} people.")
    merged = nasa_merge(df_fzpz, tlx, incl_fzpz)
    merged.to_csv(OUT_DIR / "nasa_tlx_merged.csv", index=False)
    log(f"Saved: {OUT_DIR / 'nasa_tlx_merged.csv'}")
    log(f"Overlap NASA-TLX ∩ incluibles (Fz/Pz): N={len(merged)}")
    if len(merged) > 0:
        log(merged.to_string(index=False))
        if len(merged) >= 3:
            for col, lab in [("ratio_high", "ratio High Load"),
                             ("ratio_delta", "ratio (High − Low)")]:
                rho, p = stats.spearmanr(merged["nasa_tlx_pct"], merged[col])
                log(f"  Spearman NASA-TLX vs {lab}: ρ={rho:+.3f}  p={fmt_p(float(p))}  (N={len(merged)})")

    # 6) Figures
    log("\n" + "=" * 72)
    log("FIGURES")
    log("=" * 72)

    # Fz/Pz primary
    paired_fzpz = extract_paired(df_fzpz, incl_fzpz,
                                 "low_cognitive_load", "high_cognitive_load")
    if len(paired_fzpz) > 0:
        plot_paired_slope(paired_fzpz,
                          title="Paired change Low → High  (Fz/Pz, includables)",
                          ylabel="Cognitive Load Ratio (Theta/Alpha)",
                          name="01_paired_slope_fzpz")
        plot_paired_slope_log(paired_fzpz,
                              title="Paired change Low → High  [log scale]  (Fz/Pz)",
                              name="02_paired_slope_fzpz_log")
        plot_delta_forest(paired_fzpz,
                          title="Individual deltas (High − Low) — Fz/Pz",
                          name="03_delta_forest_fzpz")
        plot_boxplot(df_fzpz, incl_fzpz,
                     title="Ratio distribution by phase (Fz/Pz, includables)",
                     name="04_boxplot_fzpz")
        log("Saved Fz/Pz figures 01–04")

    # Regions
    if df_region is not None and len(incl_region) > 0:
        paired_region = extract_paired(df_region, incl_region,
                                       "low_cognitive_load", "high_cognitive_load")
        if len(paired_region) > 0:
            plot_paired_slope(paired_region,
                              title="Paired change Low → High  (Regions F/P, includables)",
                              ylabel="Cognitive Load Ratio (Theta/Alpha)",
                              name="05_paired_slope_region")
            plot_paired_slope_log(paired_region,
                                  title="Paired change Low → High  [log scale]  (Regions F/P)",
                                  name="06_paired_slope_region_log")
            plot_delta_forest(paired_region,
                              title="Individual deltas (High − Low) — Regions F/P",
                              name="07_delta_forest_region")
            plot_boxplot(df_region, incl_region,
                         title="Ratio distribution by phase (Regions F/P, includables)",
                         name="08_boxplot_region")
            log("Saved Region figures 05–08")

    # NASA-TLX scatter
    if len(merged) >= 3:
        plot_nasa_scatter(merged, "ratio_high",
                          "EEG ratio — High Load (Fz/Pz)",
                          "Convergent validity — NASA-TLX vs EEG (High)",
                          name="09_nasa_vs_high")
        plot_nasa_scatter(merged, "ratio_delta",
                          "EEG ratio delta (High − Low)",
                          "Convergent validity — NASA-TLX vs EEG Δ",
                          name="10_nasa_vs_delta")
        log("Saved NASA-TLX figures 09–10")

    # 7) Compact results_summary.csv — lines ready for paper
    summary_rows = []
    for _, row in df_tests.iterrows():
        summary_rows.append({
            "dataset": row["dataset"],
            "contrast": row["contrast"],
            "n": int(row["n"]),
            "mean_phase_a": round(float(row["mean_a"]), 4) if np.isfinite(row["mean_a"]) else np.nan,
            "mean_phase_b": round(float(row["mean_b"]), 4) if np.isfinite(row["mean_b"]) else np.nan,
            "mean_diff_b_minus_a": round(float(row["mean_diff"]), 4) if np.isfinite(row["mean_diff"]) else np.nan,
            "ci95_mean_diff": f"[{row['ci_low_mean_diff']:+.3f}, {row['ci_high_mean_diff']:+.3f}]"
                              if np.isfinite(row["ci_low_mean_diff"]) else "NA",
            "wilcoxon_p": fmt_p(row["wilcoxon_p"]),
            "paired_t_log_p": fmt_p(row["ttest_log_p"]),
            "cohens_dz": round(float(row["cohens_dz"]), 3) if np.isfinite(row["cohens_dz"]) else np.nan,
            "rank_biserial_r": round(float(row["rank_biserial_r"]), 3) if np.isfinite(row["rank_biserial_r"]) else np.nan,
            "signs_b_gt_a": f"{int(row['n_b_greater_a'])}/{int(row['n'])}",
            "sign_test_p": fmt_p(row["sign_p"]),
            "alternative": row["alternative"],
        })
    pd.DataFrame(summary_rows).to_csv(OUT_DIR / "results_summary.csv", index=False)
    log(f"\nSaved: {OUT_DIR / 'results_summary.csv'}")
    log("\nDONE.")


if __name__ == "__main__":
    try:
        main()
    finally:
        log.close()
