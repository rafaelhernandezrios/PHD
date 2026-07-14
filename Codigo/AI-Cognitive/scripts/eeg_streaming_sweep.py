"""Sweep all subjects with the honest streaming protocol.

Uses 50% of each subject's 'normal' windows as calibration, rest as held-out test.
Trains hierarchical RF on the other 14 subjects (themselves calibrated against
their own full 'normal' set).
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score

CSV = "csv/eeg_window_features.csv"
META = {"file_name", "subject_id", "task_type", "condition_4", "load_3", "load_2",
        "timestamp", "window_start_idx", "window_end_idx"}


def cols_of(df):
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


def safe_zscore(X, mu, sd):
    Z = (X - mu) / sd
    Z = np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(Z, -50.0, 50.0)


def train(df_cal, cols):
    X1 = fmat(df_cal, cols)
    y1 = df_cal["load_2"].astype(str).to_numpy()
    sc1 = StandardScaler().fit(X1)
    c1 = RandomForestClassifier(n_estimators=400, class_weight="balanced",
                                n_jobs=-1, random_state=42)
    c1.fit(sc1.transform(X1), y1)
    task = df_cal[df_cal["load_3"].isin(["low", "high"])]
    X2 = fmat(task, cols)
    y2 = task["load_3"].astype(str).to_numpy()
    sc2 = StandardScaler().fit(X2)
    c2 = RandomForestClassifier(n_estimators=400, class_weight="balanced",
                                n_jobs=-1, random_state=42)
    c2.fit(sc2.transform(X2), y2)
    return (sc1, c1), (sc2, c2)


def predict(X, s1, s2):
    sc1, c1 = s1; sc2, c2 = s2
    p1 = c1.predict_proba(sc1.transform(X))
    cls1 = list(c1.classes_)
    p_norm = p1[:, cls1.index("normal")]
    p2 = c2.predict_proba(sc2.transform(X))
    cls2 = list(c2.classes_)
    p_high_given = p2[:, cls2.index("high")]
    p_task = 1 - p_norm
    proba = np.stack([p_norm, p_task * (1 - p_high_given), p_task * p_high_given], axis=1)
    return np.array(["normal", "low", "high"])[np.argmax(proba, axis=1)]


def main():
    df = pd.read_csv(CSV)
    cols = cols_of(df)
    subjects = sorted(df.subject_id.unique())
    results = []
    for s in subjects:
        test = df[df.subject_id == s]
        train_raw = df[df.subject_id != s]

        # held-out subject: split normal windows 50/50 for calib vs test
        norm = test[test.load_3 == "normal"]
        n_cal = max(5, len(norm) // 2)
        baseline = norm.head(n_cal)
        test_stream = test.drop(baseline.index)

        mu_te, sd_te = calib(baseline, cols)

        # calibrate every training subject against their own full normal windows
        parts = []
        for sj, g in train_raw.groupby("subject_id"):
            b = g[g.load_3 == "normal"]
            mu_s, sd_s = calib(b if len(b) >= 5 else g, cols)
            gc = g.copy()
            gc[cols] = safe_zscore(fmat(g, cols), mu_s, sd_s)
            parts.append(gc)
        train_cal = pd.concat(parts, ignore_index=True)

        s1, s2 = train(train_cal, cols)

        Xt = safe_zscore(fmat(test_stream, cols), mu_te, sd_te)
        pred = predict(Xt, s1, s2)
        y3 = test_stream.load_3.astype(str).to_numpy()
        y2 = test_stream.load_2.astype(str).to_numpy()
        pred2 = np.where(pred == "normal", "normal", "alta")
        f1_bin = f1_score(y2, pred2, average="macro", zero_division=0)
        acc_bin = (pred2 == y2).mean()
        f1_3 = f1_score(y3, pred, average="macro", zero_division=0)
        acc_3 = (pred == y3).mean()
        results.append((s, len(test_stream), acc_bin, f1_bin, acc_3, f1_3))
        print(f"  subj {s:2d}: n={len(test_stream):4d}  bin_acc={acc_bin:.3f} bin_f1={f1_bin:.3f}  "
              f"3cls_acc={acc_3:.3f} 3cls_f1={f1_3:.3f}")

    print("\n=== Streaming-honest summary ===")
    res = np.array([[r[2], r[3], r[4], r[5]] for r in results])
    print(f"Binary  : acc={res[:,0].mean():.3f}±{res[:,0].std():.3f}  "
          f"macro-f1={res[:,1].mean():.3f}±{res[:,1].std():.3f}")
    print(f"3 classes: acc={res[:,2].mean():.3f}±{res[:,2].std():.3f}  "
          f"macro-f1={res[:,3].mean():.3f}±{res[:,3].std():.3f}")


if __name__ == "__main__":
    main()
