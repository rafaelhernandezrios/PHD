"""Three-class cognitive-load classification (baja / optima / alta).

Yerkes-Dodson label mapping over condition_4:
    natural        -> 0  (baja / underload)
    low + mid      -> 1  (optima / optimal engagement)
    high           -> 2  (alta / overload)

Protocol (identical to the binary/ordinal pipeline so results are comparable):
  - per-subject z-score against each subject's own 'natural' baseline windows
  - Leave-One-Subject-Out (LOSO)
  - metrics: accuracy, macro-F1, per-class recall, pooled confusion,
    per-subject accuracy mean/std

Methods compared:
  A) RandomForest, 400 trees, class_weight='balanced'         (hard 3-class)
  B) HistGradientBoostingClassifier, class-balanced sample wts (hard 3-class)
  C) Ordinal-aware: HistGradientBoostingRegressor (MAE) on the 3 ordered
     labels, then round to {0,1,2}. Uses the ordering baja<optima<alta.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    RandomForestClassifier, HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.metrics import (
    accuracy_score, f1_score, recall_score, confusion_matrix,
)
from sklearn.utils.class_weight import compute_sample_weight

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "csv" / "eeg_window_features_aura.csv"

META = {"file_name", "subject_id", "task_type", "condition_4",
        "load_3", "load_2", "window_start_idx", "window_end_idx", "y"}

# Yerkes-Dodson 3-class mapping
YD_MAP = {"natural": 0, "normal": 0, "low": 1, "mid": 1, "high": 2}
CLASS_NAMES = ["baja", "optima", "alta"]


def feature_cols(df):
    cs = [c for c in df.columns if c not in META]
    return [c for c in cs if pd.api.types.is_numeric_dtype(df[c])]


def fmat(df, cols):
    X = df[cols].to_numpy(dtype=np.float64)
    X = np.nan_to_num(X, nan=0.0, posinf=1e10, neginf=-1e10)
    return np.clip(X, -1e10, 1e10)


def calib(g, cols):
    mu = g[cols].mean().to_numpy(dtype=np.float64)
    sd = g[cols].std().to_numpy(dtype=np.float64)
    sd = np.where(np.isfinite(sd) & (sd > 1e-9), sd, 1.0)
    return mu, sd


def safe_z(X, mu, sd):
    Z = (X - mu) / sd
    Z = np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(Z, -50.0, 50.0)


def per_subject_calibrate(df, cols):
    out = df.copy()
    for s, g in df.groupby("subject_id"):
        b = g[g.condition_4.isin(["natural", "normal"])]
        mu, sd = calib(b if len(b) >= 5 else g, cols)
        out.loc[g.index, cols] = safe_z(fmat(g, cols), mu, sd)
    return out


def per_class_recall(y_true, y_pred):
    r = recall_score(y_true, y_pred, labels=[0, 1, 2],
                     average=None, zero_division=0)
    return dict(zip(CLASS_NAMES, r))


def run_method(df_c, cols, subjects, make_model, ordinal=False):
    y_all, p_all = [], []
    per_subj_acc = []
    for s in subjects:
        tr = df_c[df_c.subject_id != s]
        te = df_c[df_c.subject_id == s]
        Xtr, ytr = fmat(tr, cols), tr["y"].to_numpy().astype(int)
        Xte, yte = fmat(te, cols), te["y"].to_numpy().astype(int)
        model = make_model()
        if ordinal:
            model.fit(Xtr, ytr.astype(float))
            pred = np.clip(np.round(model.predict(Xte)), 0, 2).astype(int)
        elif isinstance(model, HistGradientBoostingClassifier):
            sw = compute_sample_weight("balanced", ytr)
            model.fit(Xtr, ytr, sample_weight=sw)
            pred = model.predict(Xte)
        else:
            model.fit(Xtr, ytr)
            pred = model.predict(Xte)
        y_all.extend(yte); p_all.extend(pred)
        per_subj_acc.append(accuracy_score(yte, pred))
    y_all = np.array(y_all); p_all = np.array(p_all)
    return {
        "acc": accuracy_score(y_all, p_all),
        "macro_f1": f1_score(y_all, p_all, average="macro", zero_division=0),
        "recall": per_class_recall(y_all, p_all),
        "cm": confusion_matrix(y_all, p_all, labels=[0, 1, 2]),
        "subj_acc_mean": float(np.mean(per_subj_acc)),
        "subj_acc_std": float(np.std(per_subj_acc)),
        "y": y_all, "p": p_all,
    }


def run_hierarchical(df_c, cols, subjects):
    """Stage 1: baja(0) vs task(1). Stage 2 (on task): optima(1) vs alta(2)."""
    y_all, p_all, per_subj_acc = [], [], []
    for s in subjects:
        tr = df_c[df_c.subject_id != s]
        te = df_c[df_c.subject_id == s]
        Xtr, ytr = fmat(tr, cols), tr["y"].to_numpy().astype(int)
        Xte, yte = fmat(te, cols), te["y"].to_numpy().astype(int)
        # Stage 1
        y1 = (ytr > 0).astype(int)
        s1 = RandomForestClassifier(n_estimators=120,
                                    class_weight="balanced",
                                    n_jobs=-1, random_state=42).fit(Xtr, y1)
        # Stage 2 trained only on task windows (optima/alta)
        mask = ytr > 0
        s2 = RandomForestClassifier(n_estimators=120,
                                    class_weight="balanced",
                                    n_jobs=-1, random_state=42).fit(
            Xtr[mask], ytr[mask])
        p1 = s1.predict(Xte)
        pred = np.zeros(len(Xte), dtype=int)
        task_idx = p1 == 1
        if task_idx.any():
            pred[task_idx] = s2.predict(Xte[task_idx])
        y_all.extend(yte); p_all.extend(pred)
        per_subj_acc.append(accuracy_score(yte, pred))
    y_all = np.array(y_all); p_all = np.array(p_all)
    return {
        "acc": accuracy_score(y_all, p_all),
        "macro_f1": f1_score(y_all, p_all, average="macro", zero_division=0),
        "recall": per_class_recall(y_all, p_all),
        "cm": confusion_matrix(y_all, p_all, labels=[0, 1, 2]),
        "subj_acc_mean": float(np.mean(per_subj_acc)),
        "subj_acc_std": float(np.std(per_subj_acc)),
        "y": y_all, "p": p_all,
    }


def report(name, r):
    print(f"\n=== {name} ===")
    print(f"  accuracy      : {r['acc']:.3f}")
    print(f"  macro-F1      : {r['macro_f1']:.3f}")
    print(f"  per-class recall: " +
          "  ".join(f"{k}={v:.2f}" for k, v in r['recall'].items()))
    print(f"  per-subject acc : {r['subj_acc_mean']:.3f} "
          f"± {r['subj_acc_std']:.3f}")
    print("  confusion (rows=true, cols=pred):")
    print("           " + "  ".join(f"{n:>6s}" for n in CLASS_NAMES))
    for i, nm in enumerate(CLASS_NAMES):
        row = "  ".join(f"{v:6d}" for v in r['cm'][i])
        print(f"    {nm:7s} {row}")


METHODS = {
    "rf": ("A) RandomForest balanced (hard 3-class)",
           lambda: RandomForestClassifier(
               n_estimators=200, class_weight="balanced",
               n_jobs=-1, random_state=42), False),
    "hgb": ("B) HistGradientBoosting balanced (hard 3-class)",
            lambda: HistGradientBoostingClassifier(
                max_iter=120, learning_rate=0.08, max_leaf_nodes=31,
                min_samples_leaf=30, random_state=42), False),
    "ordinal": ("C) Ordinal HistGB -> round to 3 bins (ORDER-AWARE)",
                lambda: HistGradientBoostingRegressor(
                    loss="absolute_error", max_iter=300,
                    learning_rate=0.06, max_leaf_nodes=31,
                    min_samples_leaf=30, random_state=42), True),
}


def main():
    import sys, json
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    keys = list(METHODS) if which == "all" else [which]

    df = pd.read_csv(CSV)
    df = df[df.condition_4.isin(YD_MAP.keys())].copy()
    df["y"] = df.condition_4.map(YD_MAP).astype(int)
    cols = feature_cols(df)
    print(f"X: {df.shape}, features: {len(cols)}, subjects: "
          f"{df.subject_id.nunique()}")
    print("class distribution (windows):")
    print(df["y"].map(dict(enumerate(CLASS_NAMES))).value_counts().to_string())

    df_c = per_subject_calibrate(df, cols)
    subjects = sorted(df_c.subject_id.unique())

    out_path = ROOT / "csv" / "three_class_results.json"
    results = {}
    if out_path.exists():
        results = json.loads(out_path.read_text())

    if which == "hier":
        keys = []
        r = run_hierarchical(df_c, cols, subjects)
        report("D) Hierarchical 2-stage RF (baja|task -> optima|alta)", r)
        results["hier"] = {
            "name": "D) Hierarchical 2-stage RF", "acc": r["acc"],
            "macro_f1": r["macro_f1"], "recall": r["recall"],
            "cm": r["cm"].tolist(), "subj_acc_mean": r["subj_acc_mean"],
            "subj_acc_std": r["subj_acc_std"],
        }

    for k in keys:
        name, mk, ordinal = METHODS[k]
        r = run_method(df_c, cols, subjects, mk, ordinal=ordinal)
        report(name, r)
        results[k] = {
            "name": name, "acc": r["acc"], "macro_f1": r["macro_f1"],
            "recall": r["recall"], "cm": r["cm"].tolist(),
            "subj_acc_mean": r["subj_acc_mean"],
            "subj_acc_std": r["subj_acc_std"],
        }

    maj = int(pd.Series(df_c["y"]).mode()[0])
    base_acc = accuracy_score(df_c["y"], np.full(len(df_c), maj))
    results["_majority_baseline_acc"] = float(base_acc)
    results["_class_dist"] = df["y"].map(
        dict(enumerate(CLASS_NAMES))).value_counts().to_dict()
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nMajority-class baseline accuracy: {base_acc:.3f}")
    print(f"Wrote → {out_path}")


if __name__ == "__main__":
    main()
