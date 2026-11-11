"""LOSO with per-subject baseline calibration.

For each subject: take their 'normal' (natural baseline) windows, compute
mean+std per feature on that subject's baseline only, and z-score ALL of
that subject's windows against that baseline. This mimics what a real-time
system would do with a 30 s pre-task calibration phase.

NOTE: reads the legacy csv/eeg_window_features.csv (656 raw columns,
includes accelerometer/battery/status channels), NOT the cleaned/relabeled
eeg_window_features_aura.csv. main.tex's Table I "LOSO + per-subject
calibration" row (0.82 acc / 0.56 recall-rest) does NOT come from this
script -- it comes from eeg_channel_ablation.py's "binary_all8" run on the
clean 218-feature aura CSV. Run this script on this CSV inflates accuracy
(~0.85+) because the calibration z-score also latches onto non-EEG
accelerometer/status columns that happen to correlate with task blocks.
Kept here, with results now saved, so that fact is checkable rather than
silently lost -- see the classifier-comparison + channel-ablation JSONs for
the numbers that actually back the paper.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, f1_score

CSV = "csv/eeg_window_features.csv"
OUT = Path("csv/loso_calibrated_results.json")
META = {"file_name", "subject_id", "task_type", "condition_4", "load_3", "load_2",
        "timestamp", "window_start_idx", "window_end_idx"}


def feature_cols(df):
    cols = [c for c in df.columns if c not in META]
    return [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]


def per_subject_calibrate(df, cols):
    """Z-score each subject against their own 'normal' baseline windows."""
    out = df.copy()
    for s, g in df.groupby("subject_id"):
        baseline = g[g["load_3"] == "normal"]
        if len(baseline) < 5:
            # fallback: use whole subject
            mu = g[cols].mean()
            sd = g[cols].std().replace(0, 1)
        else:
            mu = baseline[cols].mean()
            sd = baseline[cols].std().replace(0, 1)
        idx = g.index
        out.loc[idx, cols] = (g[cols].values - mu.values) / sd.values
    return out


def features_array(df, cols):
    X = df[cols].to_numpy(dtype=np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=1e10, neginf=-1e10)
    return np.clip(X, -1e10, 1e10)


def loso(df, target, cols):
    df = df[df[target].notna()].copy()
    if target == "load_3":
        df = df[df[target] != "unknown"]
    subjects = sorted(df.subject_id.unique())
    all_y, all_pred = [], []
    per_subj = []
    for s in subjects:
        tr = df[df.subject_id != s]
        te = df[df.subject_id == s]
        Xtr = features_array(tr, cols)
        Xte = features_array(te, cols)
        ytr = tr[target].astype(str).to_numpy()
        yte = te[target].astype(str).to_numpy()
        # second-pass StandardScaler on the (already calibrated) features
        sc = StandardScaler().fit(Xtr)
        Xtr = sc.transform(Xtr); Xte = sc.transform(Xte)
        clf = RandomForestClassifier(
            n_estimators=400, class_weight="balanced", n_jobs=-1, random_state=42
        )
        clf.fit(Xtr, ytr)
        pred = clf.predict(Xte)
        all_y.extend(yte); all_pred.extend(pred)
        acc = (pred == yte).mean()
        f1 = f1_score(yte, pred, average="macro", zero_division=0)
        per_subj.append((s, acc, f1))
        print(f"  subj {s:2d}: n={len(yte):4d} acc={acc:.3f} f1={f1:.3f}")
    print(f"\n=== LOSO aggregate ({target}) ===")
    print(classification_report(all_y, all_pred, zero_division=0))
    labels = sorted(set(all_y))
    print("Confusion matrix labels:", labels)
    print(confusion_matrix(all_y, all_pred, labels=labels))
    accs = np.array([p[1] for p in per_subj])
    f1s = np.array([p[2] for p in per_subj])
    print(f"Mean per-subject acc: {accs.mean():.3f}  std: {accs.std():.3f}")
    print(f"Mean per-subject macro-f1: {f1s.mean():.3f}  std: {f1s.std():.3f}")

    pooled_acc = float(np.mean(np.array(all_y) == np.array(all_pred)))
    return {
        "target": target,
        "pooled_acc": pooled_acc,
        "macro_f1_pooled": float(f1_score(all_y, all_pred, average="macro", zero_division=0)),
        "subj_acc_mean": float(accs.mean()),
        "subj_acc_std": float(accs.std()),
        "subj_macro_f1_mean": float(f1s.mean()),
        "subj_macro_f1_std": float(f1s.std()),
    }


def main():
    df = pd.read_csv(CSV)
    cols = feature_cols(df)
    print(f"Loaded: {df.shape}, features: {len(cols)}")
    print("\n>>> Applying per-subject baseline calibration (z-score vs subject's 'normal' windows)")
    df_cal = per_subject_calibrate(df, cols)
    print("\n###### LOSO load_2 (calibrated) ######")
    r2 = loso(df_cal, "load_2", cols)
    print("\n###### LOSO load_3 (calibrated) ######")
    r3 = loso(df_cal, "load_3", cols)

    out = {"csv": CSV,
           "note": "calibrated on the legacy 656-col CSV -- NOT the source of main.tex Table I's "
                   "0.82 row, see module docstring. Kept for comparison.",
           "load_2": r2, "load_3": r3}
    OUT.write_text(json.dumps(out, indent=2))
    print("\nSaved ->", OUT)


if __name__ == "__main__":
    main()
