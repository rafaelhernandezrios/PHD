"""Per-window latency of the closed loop, measured single-window (batch size 1).

The streaming figure in eeg_realtime_infer.py divides a 50-window batch by 50,
which reports throughput, not latency: in deployment windows arrive one at a
time and the per-call overhead is paid on every one. This benchmark measures
what the loop actually incurs per window, and the UDP hop and policy step that
carry the decision into Unity.

Usage: python scripts/bench_latency.py
"""
from __future__ import annotations

import platform
import socket
import statistics as st
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

REPEATS = 500


def cpu_name() -> str:
    if platform.system() == "Darwin":
        try:
            return subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
        except Exception:
            pass
    return platform.processor() or platform.machine()


def report(label: str, samples_ms: list[float]) -> dict:
    s = sorted(samples_ms)
    n = len(s)
    out = {
        "mean": st.mean(s), "p50": s[n // 2],
        "p95": s[int(n * 0.95)], "p99": s[int(n * 0.99)], "max": s[-1],
    }
    print(f"  {label:34s} mean {out['mean']:7.3f}  p50 {out['p50']:7.3f}"
          f"  p95 {out['p95']:7.3f}  p99 {out['p99']:7.3f}  max {out['max']:7.3f}")
    return out


def main():
    print(f"CPU: {cpu_name()}   Python {platform.python_version()}   "
          f"{platform.system()} {platform.release()}")
    print(f"(all times in ms, single window, n={REPEATS})\n")

    results = {"cpu": cpu_name(), "repeats": REPEATS}

    # ---- inference: hierarchical RF, batch size 1 --------------------------
    from eeg_realtime_infer import (
        CSV, feature_cols, feature_matrix, apply_calibration,
        train_hierarchical, predict_hierarchical,
    )

    df = pd.read_csv(CSV)
    cols = feature_cols(df)
    print(f"features: {len(cols)}   windows: {len(df)}")

    # calibrate per subject against its own rest windows, then train
    parts = []
    for _, g in df.groupby("subject_id"):
        base = g[g["load_2"].astype(str) == "normal"]
        if len(base) == 0:
            continue
        mu = feature_matrix(base, cols).mean(axis=0)
        sd = feature_matrix(base, cols).std(axis=0)
        gc = g.copy()
        gc[cols] = apply_calibration(feature_matrix(g, cols), mu, sd)
        parts.append(gc)
    train_cal = pd.concat(parts, ignore_index=True)

    print("training hierarchical RF (stage1 + stage2)...", end=" ", flush=True)
    t0 = time.perf_counter()
    stage1, stage2 = train_hierarchical(train_cal, cols)
    print(f"{time.perf_counter()-t0:.1f}s\n")

    X = feature_matrix(train_cal.head(REPEATS + 50), cols)
    mu = X[:50].mean(axis=0)
    sd = X[:50].std(axis=0)
    # same guard the pipeline uses: constant features keep their raw value
    sd = np.where(sd < 1e-9, 1.0, sd)

    print("component latencies")

    one = X[50:51]
    predict_hierarchical(apply_calibration(one, mu, sd), stage1, stage2)  # warm up

    lat = []
    for i in range(REPEATS):
        w = X[50 + i: 51 + i]
        t0 = time.perf_counter()
        apply_calibration(w, mu, sd)
        lat.append((time.perf_counter() - t0) * 1e3)
    results["calibration"] = report("per-subject calibration", lat)

    lat = []
    for i in range(REPEATS):
        w = apply_calibration(X[50 + i: 51 + i], mu, sd)
        t0 = time.perf_counter()
        predict_hierarchical(w, stage1, stage2)
        lat.append((time.perf_counter() - t0) * 1e3)
    results["inference_batch1"] = report("hierarchical RF inference (b=1)", lat)

    # the batched number, for comparison with the earlier reporting
    B = 50
    lat = []
    for i in range(REPEATS // 10):
        w = apply_calibration(X[50: 50 + B], mu, sd)
        t0 = time.perf_counter()
        predict_hierarchical(w, stage1, stage2)
        lat.append((time.perf_counter() - t0) * 1e3 / B)
    results["inference_batch50_per_window"] = report(
        f"same, batched b={B} (per window)", lat)

    # ---- policy step -------------------------------------------------------
    from adaptive_replay import AdaptivePolicy, PolicyConfig
    pol = AdaptivePolicy(PolicyConfig())
    rng = np.random.default_rng(0)
    lat = []
    for i in range(REPEATS):
        v = float(rng.uniform(0, 2.5))
        t0 = time.perf_counter()
        pol.step(i * 2.0, v)
        lat.append((time.perf_counter() - t0) * 1e3)
    results["policy_step"] = report("adaptive policy step", lat)

    # ---- UDP hop to Unity --------------------------------------------------
    rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rx.bind(("127.0.0.1", 0))
    port = rx.getsockname()[1]
    rx.settimeout(1.0)
    tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    lat = []
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        tx.sendto(b"1.372", ("127.0.0.1", port))
        rx.recv(64)
        lat.append((time.perf_counter() - t0) * 1e3)
    rx.close()
    tx.close()
    results["udp_hop"] = report("UDP hop to Unity (loopback)", lat)

    # ---- budget ------------------------------------------------------------
    per_window = (results["calibration"]["p95"]
                  + results["inference_batch1"]["p95"]
                  + results["policy_step"]["p95"]
                  + results["udp_hop"]["p95"])
    stride_ms = 2000.0
    print(f"\np95 budget per window (calib + infer + policy + UDP): "
          f"{per_window:.2f} ms  =  {per_window / stride_ms * 100:.2f}% of the "
          f"{stride_ms:.0f} ms stride")
    print("Feature extraction is measured separately by the windowing script; "
          "it is the dominant term and is reported alongside these numbers.")
    results["p95_budget_ms"] = per_window
    results["stride_ms"] = stride_ms

    import json
    out = ROOT / "csv" / "latency_bench.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\n[json] {out}")


if __name__ == "__main__":
    main()
