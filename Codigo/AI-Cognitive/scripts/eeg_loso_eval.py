"""Leave-One-Subject-Out evaluation to detect subject leakage.

NOTE: reads the legacy csv/eeg_window_features.csv (656 raw columns,
includes accelerometer/battery/status channels) -- this is deliberate:
it reproduces the paper's Table I "LOSO, no calibration" row (main.tex),
which was computed on this same file. Do not silently repoint this at
eeg_window_features_aura.csv without also updating what Table I claims.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, f1_score, recall_score

CSV = "csv/eeg_window_features.csv"
OUT = Path("csv/loso_eval_results.json")
META = {"file_name", "subject_id", "task_type", "condition_4", "load_3", "load_2", "timestamp"}


def features(df):
    cols = [c for c in df.columns if c not in META]
    cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    X = df[cols].to_numpy(dtype=np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=1e10, neginf=-1e10)
    X = np.clip(X, -1e10, 1e10)
    return X, cols


def loso(df, target):
    df = df[df[target].notna()].copy()
    if target == "load_3":
        df = df[df[target] != "unknown"]
    subjects = sorted(df.subject_id.unique())
    all_y, all_pred = [], []
    per_subj = []
    for s in subjects:
        tr = df[df.subject_id != s]
        te = df[df.subject_id == s]
        Xtr, _ = features(tr); Xte, _ = features(te)
        ytr = tr[target].astype(str).to_numpy()
        yte = te[target].astype(str).to_numpy()
        sc = StandardScaler().fit(Xtr)
        Xtr = sc.transform(Xtr); Xte = sc.transform(Xte)
        clf = RandomForestClassifier(
            n_estimators=300, class_weight="balanced", n_jobs=-1, random_state=42
        )
        clf.fit(Xtr, ytr)
        pred = clf.predict(Xte)
        all_y.extend(yte); all_pred.extend(pred)
        per_subj.append((s, len(yte), f1_score(yte, pred, average="macro", zero_division=0),
                         (pred == yte).mean()))
        print(f"  subj {s:2d}: n={len(yte):4d} acc={(pred==yte).mean():.3f} f1={per_subj[-1][2]:.3f}")
    print(f"\n=== LOSO aggregate for {target} ===")
    print(classification_report(all_y, all_pred, zero_division=0))
    print("Confusion matrix:")
    labels = sorted(set(all_y))
    print(labels)
    print(confusion_matrix(all_y, all_pred, labels=labels))
    accs = np.array([p[3] for p in per_subj])
    f1s = np.array([p[2] for p in per_subj])
    print(f"\nMean per-subject acc: {accs.mean():.3f}  std: {accs.std():.3f}")
    print(f"Mean per-subject macro-f1: {f1s.mean():.3f}")

    pooled_acc = float(np.mean(np.array(all_y) == np.array(all_pred)))
    rec = {}
    for lab in labels:
        rec[lab] = float(recall_score(all_y, all_pred, labels=[lab], average="micro", zero_division=0))
    return {
        "target": target,
        "pooled_acc": pooled_acc,
        "macro_f1_pooled": float(f1_score(all_y, all_pred, average="macro", zero_division=0)),
        "recall_by_label": rec,
        "subj_acc_mean": float(accs.mean()),
        "subj_acc_std": float(accs.std()),
        "subj_macro_f1_mean": float(f1s.mean()),
    }


def main():
    df = pd.read_csv(CSV)
    print("Loaded:", df.shape, "subjects:", df.subject_id.nunique())
    print("\n###### LOSO load_2 ######")
    r2 = loso(df, "load_2")
    print("\n###### LOSO load_3 ######")
    r3 = loso(df, "load_3")

    out = {"csv": CSV, "note": "no calibration -- reproduces main.tex Table I 'LOSO, no calibration' row",
           "load_2": r2, "load_3": r3}
    OUT.write_text(json.dumps(out, indent=2))
    print("\nSaved ->", OUT)


if __name__ == "__main__":
    main()
