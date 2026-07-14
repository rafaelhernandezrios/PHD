"""Streaming with online baseline recalibration.

Idea: the per-subject baseline (mu, sd of every feature on the subject's
'normal' windows) can drift within a session. We start from the initial
calibration windows, then update mu/sd via EMA every time a new window is
predicted as 'normal' with confidence above a threshold.

Two updates are tracked:
    mu      <- (1-alpha) * mu      + alpha * x          (EMA mean)
    sq_mean <- (1-alpha) * sq_mean + alpha * x**2       (EMA of x^2)
    sd      = sqrt(max(sq_mean - mu**2, eps))

A window is z-scored with the CURRENT mu/sd, then classified. If the
predicted 'normal' probability is >= conf_thresh, mu/sq_mean are updated
with the raw features (not the z-scored ones).

This mimics what a real-time BCI would do: it never knows the true label,
but trusts the model's own high-confidence 'normal' predictions to keep
the baseline aligned with the user.

Run from the AI-Cognitive directory.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score

AS_CSV = "csv/eeg_window_features.csv"
ZYMA_CSV = "csv/zyma_window_features.csv"

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


def safe_z(X, mu, sd):
    Z = (X - mu) / sd
    Z = np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(Z, -50.0, 50.0)


def train_binary(train_cal, cols, n_est=400):
    X = fmat(train_cal, cols)
    y = train_cal["load_2"].astype(str).to_numpy()
    sc = StandardScaler().fit(X)
    clf = RandomForestClassifier(n_estimators=n_est, class_weight="balanced",
                                 n_jobs=-1, random_state=42)
    clf.fit(sc.transform(X), y)
    return sc, clf


def stream_online(test_stream_df, cols, mu0, sd0, sc, clf,
                  alpha=0.02, conf_thresh=0.75, ema_smooth=0.4):
    """Process windows one by one, updating baseline online."""
    test_stream_df = test_stream_df.sort_values(
        ["file_name", "window_start_idx"]
    ).reset_index(drop=True)
    raw = fmat(test_stream_df, cols)
    n, p = raw.shape

    mu = mu0.copy()
    sq_mean = (sd0 ** 2) + mu0 ** 2  # E[x^2] = var + mean^2
    classes = list(clf.classes_)
    idx_normal = classes.index("normal")

    preds = np.empty(n, dtype=object)
    proba_smooth = np.zeros((n, len(classes)))
    ema = None
    updates = 0
    for i in range(n):
        x = raw[i]
        sd = np.sqrt(np.maximum(sq_mean - mu ** 2, 1e-9))
        sd = np.where(np.isfinite(sd) & (sd > 1e-9), sd, 1.0)
        z = safe_z(x, mu, sd)
        z_t = sc.transform(z.reshape(1, -1))
        proba = clf.predict_proba(z_t)[0]
        if ema is None:
            ema = proba.copy()
        else:
            ema = ema_smooth * proba + (1 - ema_smooth) * ema
        proba_smooth[i] = ema
        preds[i] = classes[int(np.argmax(ema))]
        # Online update with high-confidence normal
        if proba[idx_normal] >= conf_thresh:
            mu = (1 - alpha) * mu + alpha * x
            sq_mean = (1 - alpha) * sq_mean + alpha * (x ** 2)
            updates += 1
    return preds, proba_smooth, updates


def run_dataset(csv_path, label):
    print(f"\n===== {label} =====")
    df = pd.read_csv(csv_path)
    cols = cols_of(df)
    subjects = sorted(df.subject_id.unique())
    print(f"  {df.shape}, features: {len(cols)}, subjects: {len(subjects)}")

    rows_static = []
    rows_online = []
    for s in subjects:
        test = df[df.subject_id == s]
        train_raw = df[df.subject_id != s]
        norm = test[test.load_3 == "normal"]
        n_cal = max(5, len(norm) // 2)
        baseline = norm.head(n_cal)
        test_stream = test.drop(baseline.index)
        mu_te, sd_te = calib(baseline, cols)

        # Train: calibrate each training subject against own normal
        parts = []
        for sj, g in train_raw.groupby("subject_id"):
            b = g[g.load_3 == "normal"]
            mu_s, sd_s = calib(b if len(b) >= 5 else g, cols)
            gc = g.copy()
            gc[cols] = safe_z(fmat(g, cols), mu_s, sd_s)
            parts.append(gc)
        train_cal = pd.concat(parts, ignore_index=True)
        sc, clf = train_binary(train_cal, cols)

        # ---- Variant A: static baseline (reference) ----
        Xt = safe_z(fmat(test_stream, cols), mu_te, sd_te)
        pred_static = clf.predict(sc.transform(Xt))

        # ---- Variant B: online recalibration ----
        pred_online, _, n_upd = stream_online(
            test_stream, cols, mu_te, sd_te, sc, clf,
            alpha=globals().get("_ALPHA", 0.02),
            conf_thresh=globals().get("_CONF", 0.75),
            ema_smooth=0.4,
        )
        # align order
        test_ord = test_stream.sort_values(["file_name", "window_start_idx"])
        y_true = test_ord["load_2"].astype(str).to_numpy()

        # static is in the original test_stream order; recompute with sorted order
        Xt_ord = safe_z(fmat(test_ord, cols), mu_te, sd_te)
        pred_static_ord = clf.predict(sc.transform(Xt_ord))

        acc_s = (pred_static_ord == y_true).mean()
        f1_s = f1_score(y_true, pred_static_ord, average="macro", zero_division=0)
        acc_o = (pred_online == y_true).mean()
        f1_o = f1_score(y_true, pred_online, average="macro", zero_division=0)

        rows_static.append((s, acc_s, f1_s))
        rows_online.append((s, acc_o, f1_o, n_upd, len(test_stream)))

        delta = acc_o - acc_s
        flag = " 🟢" if delta > 0.03 else (" 🔴" if delta < -0.03 else "")
        print(f"  subj {s:>3}: static acc={acc_s:.3f} f1={f1_s:.3f}  |  "
              f"online acc={acc_o:.3f} f1={f1_o:.3f}  Δ={delta:+.3f}  "
              f"updates={n_upd}/{len(test_stream)}{flag}")

    s_arr = np.array([[r[1], r[2]] for r in rows_static])
    o_arr = np.array([[r[1], r[2]] for r in rows_online])
    print(f"\n  STATIC : acc={s_arr[:,0].mean():.3f} ± {s_arr[:,0].std():.3f}  "
          f"macro-f1={s_arr[:,1].mean():.3f} ± {s_arr[:,1].std():.3f}")
    print(f"  ONLINE : acc={o_arr[:,0].mean():.3f} ± {o_arr[:,0].std():.3f}  "
          f"macro-f1={o_arr[:,1].mean():.3f} ± {o_arr[:,1].std():.3f}")
    print(f"  Δ mean acc: {(o_arr[:,0]-s_arr[:,0]).mean():+.3f}  "
          f"Δ mean f1: {(o_arr[:,1]-s_arr[:,1]).mean():+.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["as", "zyma", "both"], default="both")
    parser.add_argument("--alpha", type=float, default=0.02)
    parser.add_argument("--conf", type=float, default=0.75)
    args = parser.parse_args()
    # Patch the stream_online defaults via closure
    global _ALPHA, _CONF
    _ALPHA, _CONF = args.alpha, args.conf
    if args.dataset in ("as", "both"):
        run_dataset(AS_CSV, "Arithmetic/Stroop")
    if args.dataset in ("zyma", "both"):
        run_dataset(ZYMA_CSV, "Zyma PhysioNet")


if __name__ == "__main__":
    main()
