"""Channel-montage ablation + physiological Cognitive-Load Index (CLI).

Motivation (literature): frontal-midline theta (Fz) rises and parietal
alpha (Pz) falls with mental workload; the minimum reliable montage is
Fz+Pz, and frontal-alpha asymmetry (FAA) reflects affect, not load.
We therefore (i) drop the FAA asymmetry features, (ii) compare three
montages, and (iii) report the canonical index CLI = theta(Fz)/alpha(Pz).

Usage:
    python eeg_channel_ablation.py binary  all8|fp6|fzpz2
    python eeg_channel_ablation.py hier    all8|fp6|fzpz2
    python eeg_channel_ablation.py cli
Results are appended to csv/ablation_results.json.
"""
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "csv" / "eeg_window_features_aura.csv"
OUT = ROOT / "csv" / "ablation_results.json"

META = {"file_name", "subject_id", "task_type", "condition_4",
        "load_3", "load_2", "window_start_idx", "window_end_idx", "y"}

# 8 AURA EEG channels
ALL_CH = ["Fp1", "Fp2", "F3", "Fz", "F4", "P3", "Pz", "P4"]
# frontal + parietal, dropping noisy frontopolar Fp1/Fp2
FP6 = ["Fz", "F3", "F4", "Pz", "P3", "P4"]
# minimum reliable montage
FZPZ = ["Fz", "Pz"]

# region-aggregate features (physiological, keep when >1 channel montage)
REGION = ["frontal_theta_parietal_alpha", "frontal_mean_theta",
          "parietal_mean_theta", "front_minus_parietal_theta",
          "frontal_mean_alpha", "parietal_mean_alpha",
          "front_minus_parietal_alpha", "frontal_mean_beta",
          "parietal_mean_beta", "front_minus_parietal_beta"]
# FAA asymmetry features -> always dropped
FAA = ["asym_alpha_Fp1_Fp2", "asym_beta_Fp1_Fp2", "asym_alpha_F3_F4",
       "asym_beta_F3_F4", "asym_alpha_P3_P4", "asym_beta_P3_P4"]

YD_MAP = {"natural": 0, "normal": 0, "low": 1, "mid": 1, "high": 2}
CLASS_NAMES = ["baja", "optima", "alta"]

MONTAGES = {"all8": (ALL_CH, True), "fp6": (FP6, True), "fzpz2": (FZPZ, False)}


def montage_cols(df, channels, use_region):
    cols = []
    for c in df.columns:
        if c in META or c in FAA:
            continue
        if any(c.startswith(ch + "_") for ch in channels):
            cols.append(c)
        elif use_region and c in REGION:
            cols.append(c)
    return [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]


def fmat(df, cols):
    X = df[cols].to_numpy(dtype=np.float64)
    X = np.nan_to_num(X, nan=0.0, posinf=1e10, neginf=-1e10)
    return np.clip(X, -1e10, 1e10)


def per_subject_calibrate(df, cols):
    out = df.copy()
    for s, g in df.groupby("subject_id"):
        b = g[g.condition_4.isin(["natural", "normal"])]
        base = b if len(b) >= 5 else g
        mu = base[cols].mean().to_numpy(float)
        sd = base[cols].std().to_numpy(float)
        sd = np.where(np.isfinite(sd) & (sd > 1e-9), sd, 1.0)
        Z = (fmat(g, cols) - mu) / sd
        out.loc[g.index, cols] = np.clip(np.nan_to_num(Z), -50, 50)
    return out


def load():
    df = pd.read_csv(CSV)
    df = df[df.condition_4.isin(YD_MAP)].copy()
    df["y"] = df.condition_4.map(YD_MAP).astype(int)
    return df


def run_binary(df_c, cols, subjects):
    ya, pa, sacc = [], [], []
    for s in subjects:
        tr, te = df_c[df_c.subject_id != s], df_c[df_c.subject_id == s]
        ytr = (tr["y"] > 0).astype(int).to_numpy()
        yte = (te["y"] > 0).astype(int).to_numpy()
        m = RandomForestClassifier(n_estimators=150, class_weight="balanced",
                                   n_jobs=-1, random_state=42).fit(fmat(tr, cols), ytr)
        pr = m.predict(fmat(te, cols))
        ya.extend(yte); pa.extend(pr); sacc.append(accuracy_score(yte, pr))
    ya, pa = np.array(ya), np.array(pa)
    return {"n_features": len(cols), "acc": accuracy_score(ya, pa),
            "macro_f1": f1_score(ya, pa, average="macro"),
            "recall_rest": recall_score(ya, pa, pos_label=0),
            "recall_task": recall_score(ya, pa, pos_label=1),
            "subj_acc_mean": float(np.mean(sacc)),
            "subj_acc_std": float(np.std(sacc))}


def run_hier(df_c, cols, subjects):
    ya, pa, sacc = [], [], []
    for s in subjects:
        tr, te = df_c[df_c.subject_id != s], df_c[df_c.subject_id == s]
        Xtr, ytr = fmat(tr, cols), tr["y"].to_numpy()
        Xte, yte = fmat(te, cols), te["y"].to_numpy()
        s1 = RandomForestClassifier(n_estimators=100, class_weight="balanced",
                                    n_jobs=-1, random_state=42).fit(Xtr, (ytr > 0).astype(int))
        mask = ytr > 0
        s2 = RandomForestClassifier(n_estimators=100, class_weight="balanced",
                                    n_jobs=-1, random_state=42).fit(Xtr[mask], ytr[mask])
        p1 = s1.predict(Xte)
        pred = np.zeros(len(Xte), int)
        ti = p1 == 1
        if ti.any():
            pred[ti] = s2.predict(Xte[ti])
        ya.extend(yte); pa.extend(pred); sacc.append(accuracy_score(yte, pred))
    ya, pa = np.array(ya), np.array(pa)
    rc = recall_score(ya, pa, labels=[0, 1, 2], average=None, zero_division=0)
    return {"n_features": len(cols), "acc": accuracy_score(ya, pa),
            "macro_f1": f1_score(ya, pa, average="macro", zero_division=0),
            "recall": dict(zip(CLASS_NAMES, rc.tolist())),
            "cm": confusion_matrix(ya, pa, labels=[0, 1, 2]).tolist(),
            "subj_acc_mean": float(np.mean(sacc)),
            "subj_acc_std": float(np.std(sacc))}


def run_cli(df):
    """CLI = theta(Fz)/alpha(Pz), z-scored vs each subject's rest baseline.
    Report Spearman rho vs the 4-level ordinal condition."""
    ord_map = {"normal": 0, "low": 1, "mid": 2, "high": 3}
    df = df.copy()
    df["lvl"] = df.condition_4.map(ord_map).astype(float)
    df["cli_raw"] = df["Fz_bp_theta"] / df["Pz_bp_alpha"].replace(0, np.nan)
    # existing frontal_theta/parietal_alpha index as cross-check
    df["ftpa"] = df["frontal_theta_parietal_alpha"]
    rows_rho, rows_rho_ft = [], []
    zvals = np.full(len(df), np.nan)
    for s, g in df.groupby("subject_id"):
        b = g[g.condition_4.isin(["natural", "normal"])]["cli_raw"]
        mu, sd = b.mean(), b.std()
        sd = sd if (np.isfinite(sd) and sd > 1e-9) else 1.0
        z = (g["cli_raw"] - mu) / sd
        zvals[[df.index.get_loc(i) for i in g.index]] = z.to_numpy()
        rho, _ = spearmanr(g["lvl"], g["cli_raw"])
        rho2, _ = spearmanr(g["lvl"], g["ftpa"])
        rows_rho.append(rho); rows_rho_ft.append(rho2)
    df["cli_z"] = zvals
    rho_all, _ = spearmanr(df["lvl"], df["cli_raw"], nan_policy="omit")
    rho_ft_all, _ = spearmanr(df["lvl"], df["ftpa"], nan_policy="omit")
    return {
        "cli_definition": "theta(Fz)/alpha(Pz)",
        "spearman_pooled": float(rho_all),
        "spearman_persubj_mean": float(np.nanmean(rows_rho)),
        "spearman_persubj_std": float(np.nanstd(rows_rho)),
        "ftpa_spearman_pooled": float(rho_ft_all),
        "ftpa_spearman_persubj_mean": float(np.nanmean(rows_rho_ft)),
        "n_windows": int(df["cli_raw"].notna().sum()),
    }


def save(key, val):
    d = json.loads(OUT.read_text()) if OUT.exists() else {}
    d[key] = val
    OUT.write_text(json.dumps(d, indent=2))


def main():
    mode = sys.argv[1]
    df = load()
    subjects = sorted(df.subject_id.unique())

    if mode == "cli":
        r = run_cli(df)
        print(json.dumps(r, indent=2)); save("cli", r); return

    mont = sys.argv[2]
    channels, use_region = MONTAGES[mont]
    cols = montage_cols(df, channels, use_region)
    df_c = per_subject_calibrate(df, cols)
    print(f"montage={mont} channels={channels} features={len(cols)}")
    if mode == "binary":
        r = run_binary(df_c, cols, subjects)
        save(f"binary_{mont}", r)
    else:
        r = run_hier(df_c, cols, subjects)
        save(f"hier_{mont}", r)
    print(json.dumps(r, indent=2))
    print("saved →", OUT)


if __name__ == "__main__":
    main()
