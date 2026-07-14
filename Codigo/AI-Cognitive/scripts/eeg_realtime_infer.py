"""Real-time inference prototype.

Pipeline:
  1) Calibration phase: collect ~30 s of 'normal' baseline windows from the user;
     compute per-feature mean/std of that subject.
  2) Streaming phase: for each new 4-s window (50% overlap → one prediction every 2 s),
     extract features, z-score with the subject's baseline, run inference, and
     apply EMA smoothing on class probabilities.
  3) Hierarchical model: first decide normal vs task; if task, decide low vs high.

This script trains the two RF stages on the existing feature CSV (using all
subjects except a held-out one for demo) and then *simulates* streaming over
the held-out subject's windows, reporting smoothed predictions.
"""
from __future__ import annotations

import argparse
import time
from collections import deque
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

CSV = "csv/eeg_window_features.csv"
META = {"file_name", "subject_id", "task_type", "condition_4", "load_3", "load_2",
        "timestamp", "window_start_idx", "window_end_idx"}


def feature_cols(df: pd.DataFrame) -> list[str]:
    cols = [c for c in df.columns if c not in META]
    return [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]


def calibrate(baseline_df: pd.DataFrame, cols: list[str]) -> Tuple[np.ndarray, np.ndarray]:
    """Compute per-feature mean/std from a subject's baseline windows."""
    mu = baseline_df[cols].mean().to_numpy(dtype=np.float64)
    sd = baseline_df[cols].std().replace(0, 1).to_numpy(dtype=np.float64)
    return mu, sd


def apply_calibration(X: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    return (X.astype(np.float64) - mu) / sd


def feature_matrix(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    X = df[cols].to_numpy(dtype=np.float64)
    X = np.nan_to_num(X, nan=0.0, posinf=1e10, neginf=-1e10)
    return np.clip(X, -1e10, 1e10)


def train_hierarchical(df_train_cal: pd.DataFrame, cols: list[str]):
    """Two-stage classifier on calibrated training data."""
    # Stage 1: normal vs task (alta)
    y1 = df_train_cal["load_2"].astype(str).to_numpy()
    X1 = feature_matrix(df_train_cal, cols)
    sc1 = StandardScaler().fit(X1)
    clf1 = RandomForestClassifier(
        n_estimators=400, class_weight="balanced", n_jobs=-1, random_state=42
    )
    clf1.fit(sc1.transform(X1), y1)

    # Stage 2: within task → low vs high (drop normal/unknown)
    task_df = df_train_cal[df_train_cal["load_3"].isin(["low", "high"])].copy()
    y2 = task_df["load_3"].astype(str).to_numpy()
    X2 = feature_matrix(task_df, cols)
    sc2 = StandardScaler().fit(X2)
    clf2 = RandomForestClassifier(
        n_estimators=400, class_weight="balanced", n_jobs=-1, random_state=42
    )
    clf2.fit(sc2.transform(X2), y2)
    return (sc1, clf1), (sc2, clf2)


def predict_hierarchical(X: np.ndarray, stage1, stage2, normal_thresh: float = 0.5):
    sc1, clf1 = stage1
    sc2, clf2 = stage2
    proba1 = clf1.predict_proba(sc1.transform(X))
    classes1 = list(clf1.classes_)
    p_normal = proba1[:, classes1.index("normal")] if "normal" in classes1 else 1 - proba1[:, 0]
    proba2 = clf2.predict_proba(sc2.transform(X))
    classes2 = list(clf2.classes_)
    p_high_given_task = proba2[:, classes2.index("high")] if "high" in classes2 else proba2[:, 0]
    # Joint distribution over {normal, low, high}
    p_task = 1 - p_normal
    p_high = p_task * p_high_given_task
    p_low = p_task * (1 - p_high_given_task)
    proba = np.stack([p_normal, p_low, p_high], axis=1)
    labels = np.array(["normal", "low", "high"])
    preds = labels[np.argmax(proba, axis=1)]
    return preds, proba, labels


def stream_subject(subj_df: pd.DataFrame, mu, sd, stage1, stage2, cols,
                   ema_alpha: float = 0.4):
    """Simulate streaming: process windows in temporal order with EMA on probabilities."""
    subj_df = subj_df.sort_values(["file_name", "window_start_idx"]).reset_index(drop=True)
    Xraw = feature_matrix(subj_df, cols)
    Xcal = apply_calibration(Xraw, mu, sd)
    _, proba_raw, labels = predict_hierarchical(Xcal, stage1, stage2)

    smoothed = np.zeros_like(proba_raw)
    ema = proba_raw[0].copy()
    inference_times = []
    for i, p in enumerate(proba_raw):
        t0 = time.perf_counter()
        ema = ema_alpha * p + (1 - ema_alpha) * ema
        smoothed[i] = ema
        inference_times.append((time.perf_counter() - t0) * 1000.0)
    preds = labels[np.argmax(smoothed, axis=1)]
    return preds, smoothed, labels, np.mean(inference_times)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--held-out", type=int, default=8,
                        help="Subject id used as the streaming target.")
    parser.add_argument("--baseline-fraction", type=float, default=0.5,
                        help="Fraction of the held-out subject's 'normal' windows used for calibration.")
    parser.add_argument("--ema-alpha", type=float, default=0.4)
    args = parser.parse_args()

    df = pd.read_csv(CSV)
    cols = feature_cols(df)
    print(f"Loaded: {df.shape}, features: {len(cols)}")

    test = df[df.subject_id == args.held_out].copy()
    train = df[df.subject_id != args.held_out].copy()
    print(f"Held-out subject {args.held_out}: {len(test)} windows. Train: {len(train)} windows.")

    # ---- Calibrate using ONLY the held-out subject's baseline windows ----
    baseline_pool = test[test["load_3"] == "normal"]
    n_cal = max(5, int(len(baseline_pool) * args.baseline_fraction))
    baseline = baseline_pool.head(n_cal)
    print(f"Calibration: {len(baseline)} 'normal' windows from subject {args.held_out}")
    mu, sd = calibrate(baseline, cols)

    # The streaming test set: everything in held-out EXCEPT the calibration windows
    test_stream = test.drop(baseline.index)

    # ---- Apply calibration to training set (per subject, using own normal windows) ----
    train_cal_parts = []
    for s, g in train.groupby("subject_id"):
        b = g[g["load_3"] == "normal"]
        mu_s, sd_s = calibrate(b if len(b) >= 5 else g, cols)
        Xc = apply_calibration(feature_matrix(g, cols), mu_s, sd_s)
        gc = g.copy()
        gc[cols] = Xc
        train_cal_parts.append(gc)
    train_cal = pd.concat(train_cal_parts, ignore_index=True)
    print(f"Calibrated training set: {len(train_cal)} windows")

    print("\nTraining hierarchical classifier (RF stage1 + RF stage2)...")
    t0 = time.perf_counter()
    stage1, stage2 = train_hierarchical(train_cal, cols)
    print(f"  trained in {time.perf_counter()-t0:.1f}s")

    print(f"\nStreaming over held-out subject {args.held_out} ({len(test_stream)} windows)...")
    preds, smoothed, labels, infer_ms = stream_subject(
        test_stream, mu, sd, stage1, stage2, cols, ema_alpha=args.ema_alpha
    )

    y_true_3 = test_stream["load_3"].astype(str).to_numpy()
    y_true_2 = test_stream["load_2"].astype(str).to_numpy()

    # Map preds → binary for load_2 comparison
    preds_2 = np.where(preds == "normal", "normal", "alta")

    print("\n=== Real-time prediction (EMA smoothed) — 3 classes ===")
    print(classification_report(y_true_3, preds, labels=["normal", "low", "high"], zero_division=0))
    print("Confusion (rows=true normal,low,high; cols=pred):")
    print(confusion_matrix(y_true_3, preds, labels=["normal", "low", "high"]))

    print("\n=== Real-time prediction — binary (normal vs alta) ===")
    print(classification_report(y_true_2, preds_2, labels=["normal", "alta"], zero_division=0))

    print(f"\nEMA step latency: {infer_ms:.3f} ms (smoothing only)")
    # A timing estimate for full inference:
    t0 = time.perf_counter()
    _, _, _ = predict_hierarchical(
        apply_calibration(feature_matrix(test_stream.head(50), cols), mu, sd),
        stage1, stage2,
    )
    print(f"Full hierarchical infer (50 windows): {(time.perf_counter()-t0)*1000:.1f} ms "
          f"→ {(time.perf_counter()-t0)*1000/50:.2f} ms/window")


if __name__ == "__main__":
    main()
