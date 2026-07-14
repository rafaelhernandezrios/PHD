"""Ordinal regression on AURA cleaned features.

Target: condition_4 mapped to {normal:0, low:1, mid:2, high:3}.
Pipeline: per-subject z-score against own 'normal' baseline, LOSO,
LightGBM regressor with MAE objective. Reports:
  - MAE (ordinal-level error)
  - Spearman rho and Pearson r
  - % predictions within ±1 ordinal level
  - Confusion matrix of rounded predictions vs. true levels
  - Comparison to two trivial baselines: predict-mean and majority-class
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error, r2_score, confusion_matrix,
)
from scipy.stats import spearmanr, pearsonr
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "csv" / "eeg_window_features_aura.csv"

META = {"file_name", "subject_id", "task_type", "condition_4",
        "load_3", "load_2", "window_start_idx", "window_end_idx", "y"}

LEVEL_MAP = {"normal": 0, "low": 1, "mid": 2, "high": 3}
LEVEL_NAMES = ["normal", "low", "mid", "high"]


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
        b = g[g.condition_4 == "normal"]
        mu, sd = calib(b if len(b) >= 5 else g, cols)
        out.loc[g.index, cols] = safe_z(fmat(g, cols), mu, sd)
    return out


def evaluate(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rho, _ = spearmanr(y_true, y_pred)
    r, _ = pearsonr(y_true, y_pred)
    within1 = float(np.mean(np.abs(y_pred - y_true) <= 1.0))
    r2 = r2_score(y_true, y_pred)
    return {
        "MAE": float(mae),
        "Spearman_rho": float(rho),
        "Pearson_r": float(r),
        "within_pm1": within1,
        "R2": float(r2),
    }


def main():
    df = pd.read_csv(CSV)
    df = df[df.condition_4.isin(LEVEL_MAP.keys())].copy()
    df["y"] = df.condition_4.map(LEVEL_MAP).astype(float)
    cols = feature_cols(df)
    print(f"X: {df.shape}, features: {len(cols)}")

    df_c = per_subject_calibrate(df, cols)
    subjects = sorted(df_c.subject_id.unique())

    y_all, p_all, subj_all = [], [], []
    per_subj = []
    for s in subjects:
        tr = df_c[df_c.subject_id != s]
        te = df_c[df_c.subject_id == s]
        Xtr, ytr = fmat(tr, cols), tr["y"].to_numpy()
        Xte, yte = fmat(te, cols), te["y"].to_numpy()
        model = HistGradientBoostingRegressor(
            loss="absolute_error",  # MAE objective; robust ordinal
            max_iter=600, learning_rate=0.05, max_leaf_nodes=63,
            min_samples_leaf=20, random_state=42,
        )
        model.fit(Xtr, ytr)
        pred = model.predict(Xte)
        y_all.extend(yte); p_all.extend(pred); subj_all.extend([s] * len(yte))
        m = evaluate(yte, pred)
        per_subj.append((s, m))
        print(f"  subj {s:2d}: n={len(yte):4d}  MAE={m['MAE']:.3f}  "
              f"ρ={m['Spearman_rho']:+.3f}  ±1={m['within_pm1']:.2f}")

    y_all = np.array(y_all); p_all = np.array(p_all)
    print("\n=== Pooled metrics ===")
    overall = evaluate(y_all, p_all)
    for k, v in overall.items():
        print(f"  {k:14s}: {v:+.3f}")

    print("\nPer-subject MAE: "
          f"mean={np.mean([m['MAE'] for _, m in per_subj]):.3f}  "
          f"std={np.std([m['MAE'] for _, m in per_subj]):.3f}")
    print("Per-subject Spearman ρ: "
          f"mean={np.mean([m['Spearman_rho'] for _, m in per_subj]):+.3f}  "
          f"std={np.std([m['Spearman_rho'] for _, m in per_subj]):.3f}")

    # ---- Baselines ----
    print("\n=== Trivial baselines ===")
    bl_mean = np.full_like(y_all, y_all.mean())
    bl_med = np.full_like(y_all, np.median(y_all))
    for name, b in [("predict-mean", bl_mean), ("predict-median", bl_med)]:
        m = evaluate(y_all, b)
        print(f"  {name:15s}: MAE={m['MAE']:.3f}  ρ={m['Spearman_rho']:+.3f}  ±1={m['within_pm1']:.2f}")

    # ---- Confusion of rounded predictions ----
    rounded = np.clip(np.round(p_all), 0, 3).astype(int)
    true_int = y_all.astype(int)
    cm = confusion_matrix(true_int, rounded, labels=[0, 1, 2, 3])
    print("\nRounded-prediction confusion (rows=true, cols=pred):")
    print("       " + "  ".join(f"{n:>6s}" for n in LEVEL_NAMES))
    for i, name in enumerate(LEVEL_NAMES):
        row = "  ".join(f"{v:6d}" for v in cm[i])
        print(f"  {name:7s} {row}")

    # ---- Save predictions for figures ----
    out = ROOT / "csv" / "ordinal_predictions.csv"
    pd.DataFrame({"subject_id": subj_all, "y_true": y_all,
                  "y_pred": p_all}).to_csv(out, index=False)
    print(f"\nWrote predictions → {out}")


if __name__ == "__main__":
    main()
