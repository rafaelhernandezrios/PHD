"""Honest streaming sweep on the Zyma dataset.

Same protocol as eeg_streaming_sweep.py:
  - LOSO across the 36 Zyma subjects.
  - Each subject calibrates with 50% of their own 'normal' windows.
  - Per-subject z-score in train (against own normal windows).
  - RandomForest, binary load_2 (normal vs alta).
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score

CSV = "csv/zyma_window_features.csv"
META = {"file_name", "subject_id", "task_type", "condition_4", "load_3", "load_2",
        "window_start_idx", "window_end_idx"}


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


def safe_z(X, mu, sd):
    Z = (X - mu) / sd
    Z = np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(Z, -50.0, 50.0)


def main():
    df = pd.read_csv(CSV)
    cols = cols_of(df)
    print(f"Zyma: {df.shape}, features: {len(cols)}")
    subjects = sorted(df.subject_id.unique())
    results = []
    for s in subjects:
        test = df[df.subject_id == s]
        train_raw = df[df.subject_id != s]
        norm = test[test.load_3 == "normal"]
        n_cal = max(5, len(norm) // 2)
        baseline = norm.head(n_cal)
        test_stream = test.drop(baseline.index)
        mu_te, sd_te = calib(baseline, cols)

        parts = []
        for sj, g in train_raw.groupby("subject_id"):
            b = g[g.load_3 == "normal"]
            mu_s, sd_s = calib(b if len(b) >= 5 else g, cols)
            gc = g.copy()
            gc[cols] = safe_z(fmat(g, cols), mu_s, sd_s)
            parts.append(gc)
        train_cal = pd.concat(parts, ignore_index=True)

        X1 = fmat(train_cal, cols)
        y1 = train_cal["load_2"].astype(str).to_numpy()
        sc1 = StandardScaler().fit(X1)
        c1 = RandomForestClassifier(n_estimators=400, class_weight="balanced",
                                    n_jobs=-1, random_state=42)
        c1.fit(sc1.transform(X1), y1)

        Xt = safe_z(fmat(test_stream, cols), mu_te, sd_te)
        pred = c1.predict(sc1.transform(Xt))
        y_true = test_stream["load_2"].astype(str).to_numpy()
        acc = (pred == y_true).mean()
        f1 = f1_score(y_true, pred, average="macro", zero_division=0)
        recall_normal = ((pred == "normal") & (y_true == "normal")).sum() / max(1, (y_true == "normal").sum())
        recall_alta = ((pred == "alta") & (y_true == "alta")).sum() / max(1, (y_true == "alta").sum())
        results.append((s, len(test_stream), acc, f1, recall_normal, recall_alta))
        print(f"  subj {s:02d}: n={len(test_stream):4d} acc={acc:.3f} f1={f1:.3f} "
              f"rec_normal={recall_normal:.3f} rec_alta={recall_alta:.3f}")

    print("\n=== Zyma streaming-honest summary (binary) ===")
    res = np.array([[r[2], r[3], r[4], r[5]] for r in results])
    print(f"acc           : {res[:,0].mean():.3f} ± {res[:,0].std():.3f}")
    print(f"macro-f1      : {res[:,1].mean():.3f} ± {res[:,1].std():.3f}")
    print(f"recall normal : {res[:,2].mean():.3f} ± {res[:,2].std():.3f}")
    print(f"recall alta   : {res[:,3].mean():.3f} ± {res[:,3].std():.3f}")


if __name__ == "__main__":
    main()
