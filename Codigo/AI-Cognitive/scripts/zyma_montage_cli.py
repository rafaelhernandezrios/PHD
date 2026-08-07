"""Montage ablation and the theta(Fz)/alpha(Pz) index, on PhysioNet eegmat.

Why here and not on the arithmetic/Stroop dataset: that recording is an
8-channel OpenBCI montage with no parietal electrode, so neither the
frontal--parietal reduction nor the classical Fz/Pz index can be computed on
it. eegmat is a 19-channel 10--20 layout, so Fz, Pz, P3 and P4 are real, and it
brings 36 subjects instead of 15.

The cost is that eegmat has two conditions, rest and mental arithmetic, so
everything here is the binary task. There is no graded load axis to correlate
against.

Outputs
-------
  csv/zyma_montage_cli.json
  paper/figures/fig_zyma_montage_cli.pdf
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).resolve().parent.parent
CSV = ROOT / "csv" / "zyma_window_features.csv"

META = {"file_name", "subject_id", "task_type", "condition_4",
        "load_2", "load_3", "window_start_idx", "window_end_idx"}

# The montages compared. The 8-channel row mirrors the positions of the
# arithmetic/Stroop headset so the two datasets can be discussed side by side;
# the 6-channel row is the frontal--parietal set; the last is the minimal pair.
MONTAGES = {
    "19 ch (full 10-20)": None,          # every channel present
    "8 ch (frontal + parietal)": ["Fp1", "Fp2", "F3", "Fz", "F4", "P3", "Pz", "P4"],
    "6 ch (Fz,F3,F4,Pz,P3,P4)": ["Fz", "F3", "F4", "Pz", "P3", "P4"],
    "2 ch (Fz,Pz)": ["Fz", "Pz"],
}

# Frontal alpha asymmetry indexes affect, not load: dropped from every montage.
FAA = re.compile(r"asym|faa", re.I)


def feature_cols(df, channels=None):
    cols = [c for c in df.columns if c not in META and not FAA.search(c)]
    if channels is None:
        return cols
    keep = set(channels)
    return [c for c in cols if c.split("_", 1)[0] in keep]


def calibrate(df, cols):
    """Per-subject z-score against that subject's own rest windows."""
    X = df[cols].to_numpy(dtype=float)
    Z = np.empty_like(X)
    for s in df.subject_id.unique():
        i = (df.subject_id == s).to_numpy()
        base = X[i & (df.load_2 == "normal").to_numpy()]
        mu, sd = (base.mean(0), base.std(0)) if len(base) else (X[i].mean(0), X[i].std(0))
        Z[i] = (X[i] - mu) / np.where(sd < 1e-9, 1.0, sd)
    return np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0)


def loso(df, cols):
    y = df.load_2.to_numpy()
    Z = calibrate(df, cols)
    acc, f1 = [], []
    for s in df.subject_id.unique():
        tr = (df.subject_id != s).to_numpy()
        te = ~tr
        clf = RandomForestClassifier(400, class_weight="balanced",
                                     random_state=0, n_jobs=-1).fit(Z[tr], y[tr])
        p = clf.predict(Z[te])
        acc.append(accuracy_score(y[te], p))
        f1.append(f1_score(y[te], p, average="macro"))
    return float(np.mean(acc)), float(np.std(acc)), float(np.mean(f1))


def main():
    df = pd.read_csv(CSV)
    chans = sorted({c.split("_", 1)[0] for c in df.columns if c not in META})
    print(f"eegmat: {len(df)} windows, {df.subject_id.nunique()} subjects, "
          f"{len(chans)} channels")
    for m in ("Fz", "Pz", "P3", "P4"):
        assert m in chans, f"falta {m}"
    print("Fz, Pz, P3 y P4 presentes y documentados\n")

    results = {}
    print(f"  {'montage':28s}{'#feat':>7}{'acc':>16}{'macro-F1':>10}")
    for name, chs in MONTAGES.items():
        cols = feature_cols(df, chs)
        a, sd, f = loso(df, cols)
        results[name] = {"n_features": len(cols), "acc": a, "acc_std": sd, "macro_f1": f}
        print(f"  {name:28s}{len(cols):7d}   {a:.3f} +- {sd:.3f}{f:10.3f}")

    # ---- the index on its own, where Fz and Pz are real --------------------
    fz_theta, pz_alpha = "Fz_bp_theta", "Pz_bp_alpha"
    assert fz_theta in df.columns and pz_alpha in df.columns
    df["CLI"] = df[fz_theta] / (df[pz_alpha] + 1e-12)

    rest_r, task_r = [], []
    for s, g in df.groupby("subject_id"):
        b = g.loc[g.load_2 == "normal", "CLI"].median()
        t = g.loc[g.load_2 == "alta", "CLI"].median()
        if np.isfinite(b) and np.isfinite(t) and b > 0:
            rest_r.append(1.0)
            task_r.append(t / b)
    stat, p = wilcoxon(task_r, np.ones_like(task_r))
    rise = float(np.median(task_r))
    n_up = int(np.sum(np.array(task_r) > 1))
    print(f"\n  CLI = theta(Fz)/alpha(Pz), each subject against its own rest:")
    print(f"    median task/rest ratio {rise:.3f}   rises in {n_up}/{len(task_r)} subjects")
    print(f"    Wilcoxon signed-rank W={stat:.0f}, p={p:.2g}")
    results["cli"] = {"median_task_over_rest": rise, "n_up": n_up,
                      "n_subjects": len(task_r), "wilcoxon_W": float(stat),
                      "wilcoxon_p": float(p)}

    out = ROOT / "csv" / "zyma_montage_cli.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\n[json] {out}")
    make_figure(results, task_r)


def make_figure(results, task_r):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "serif", "font.size": 10,
                         "axes.labelsize": 10, "xtick.labelsize": 9,
                         "ytick.labelsize": 9, "savefig.dpi": 300,
                         "savefig.bbox": "tight"})
    fig, ax = plt.subplots(1, 2, figsize=(7.1, 2.9),
                           gridspec_kw={"width_ratios": [1.5, 1.0]})

    names = [k for k in results if k != "cli"]
    accs = [results[k]["acc"] for k in names]
    errs = [results[k]["acc_std"] for k in names]
    nfeat = [results[k]["n_features"] for k in names]
    ax[0].barh(range(len(names)), accs, xerr=errs, color="#3b6fb0",
               edgecolor="black", linewidth=0.4, capsize=2.5, height=0.6)
    for i, (a, f) in enumerate(zip(accs, [results[k]["macro_f1"] for k in names])):
        ax[0].text(a + errs[i] + 0.012, i, f"{a:.3f}", va="center", fontsize=8.5)
    ax[0].set_yticks(range(len(names)))
    ax[0].set_yticklabels([f"{n}\n({f} feat.)" for n, f in zip(names, nfeat)],
                          fontsize=8)
    ax[0].invert_yaxis()
    ax[0].set_xlim(0.55, 0.98)
    ax[0].set_xlabel("binary accuracy, LOSO + calibration")
    ax[0].set_title("(a) montage ablation", fontsize=10)

    ax[1].axhline(1.0, color="#888", linestyle=":", linewidth=0.9)
    ax[1].scatter(np.random.default_rng(0).normal(0, 0.045, len(task_r)),
                  task_r, s=22, color="#c0392b", alpha=0.75,
                  edgecolor="black", linewidth=0.4)
    ax[1].set_xlim(-0.3, 0.3)
    ax[1].set_xticks([])
    ax[1].set_ylabel(r"CLI on task / CLI at rest")
    ax[1].set_title(r"(b) $\theta$(Fz)/$\alpha$(Pz)", fontsize=10)

    fig.tight_layout()
    out = ROOT / "paper" / "figures" / "fig_zyma_montage_cli.pdf"
    fig.savefig(out)
    print(f"[fig] {out}")


if __name__ == "__main__":
    main()
