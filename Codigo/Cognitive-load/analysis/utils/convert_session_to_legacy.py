#!/usr/bin/env python3
"""
convert_session_to_legacy.py

Bridges a v2 session folder (eeg_raw.csv / eeg_filtered.csv / events.csv /
trials_stroop.csv) back to the single-file layout that the existing scripts in
analysis/ expect:

    eeg_data_<timestamp>.csv   with columns
    timestamp, phase, label, ecological_modality, channel_0 .. channel_7

Two things differ from the old recorder, on purpose:

1. Decimation to 50 Hz is done with ``scipy.signal.decimate``, which low-pass
   filters before dropping samples. The old recorder kept every 5th sample with
   no anti-alias filter, folding 38-42 Hz straight onto the 8-12 Hz alpha band
   that the cognitive-load index divides by.

2. The source is the RAW recording, re-filtered with a zero-phase filter
   (``filtfilt``). The old files could only ever contain the causal one-pass
   result, phase-distorted and impossible to undo.

Usage:
    python convert_session_to_legacy.py <session_dir> [--fs-out 50] [--keep-250]
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy import signal

CHANNELS = [f"channel_{i}" for i in range(8)]


def load_session(session_dir):
    raw_path = os.path.join(session_dir, "eeg_raw.csv")
    if not os.path.exists(raw_path):
        raise SystemExit(f"No eeg_raw.csv in {session_dir}")
    return pd.read_csv(raw_path)


def rebuild_filtered(df, fs, line_freq=60.0, band=(1.0, 40.0)):
    """Zero-phase notch + band-pass on the raw signal."""
    x = df[CHANNELS].to_numpy(dtype=float)

    # Channels that never move are dead electrodes; filtering them is
    # meaningless and filtfilt on a constant just returns the constant.
    alive = x.std(axis=0) > 1e-6

    b_notch, a_notch = signal.iirnotch(line_freq, 30.0, fs)
    nyq = fs / 2
    b_band, a_band = signal.butter(4, [band[0] / nyq, band[1] / nyq], btype="band")

    y = x.copy()
    if alive.any():
        sub = x[:, alive]
        sub = signal.filtfilt(b_notch, a_notch, sub, axis=0)
        sub = signal.filtfilt(b_band, a_band, sub, axis=0)
        y[:, alive] = sub
    y[:, ~alive] = np.nan  # make dead channels obvious instead of "very clean"
    return y, alive


def decimate_block(y, factor):
    """Anti-aliased decimation, channel by channel, NaN-safe."""
    out = np.empty((int(np.ceil(y.shape[0] / factor)), y.shape[1]))
    for c in range(y.shape[1]):
        col = y[:, c]
        if not np.isfinite(col).all():
            out[:, c] = np.nan
            continue
        d = signal.decimate(col, factor, ftype="iir", zero_phase=True)
        out[: len(d), c] = d
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session_dir")
    ap.add_argument("--fs-in", type=float, default=250.0)
    ap.add_argument("--fs-out", type=float, default=50.0)
    ap.add_argument("--line-freq", type=float, default=60.0)
    ap.add_argument("--keep-250", action="store_true",
                    help="skip decimation and emit the full-rate recording")
    args = ap.parse_args()

    df = load_session(args.session_dir)
    print(f"read {len(df)} samples from {args.session_dir}")

    y, alive = rebuild_filtered(df, args.fs_in, args.line_freq)
    dead = [CHANNELS[i] for i in range(8) if not alive[i]]
    if dead:
        print(f"WARNING dead/flat channels written as NaN: {', '.join(dead)}")

    ts = df["timestamp_lsl"].to_numpy(dtype=float)
    phase = df["phase"].to_numpy()
    label = df["label"].to_numpy()
    eco = df.get("ecological_modality", pd.Series([""] * len(df))).to_numpy()

    if args.keep_250:
        factor = 1
    else:
        factor = int(round(args.fs_in / args.fs_out))
        if abs(args.fs_in / factor - args.fs_out) > 0.01:
            raise SystemExit(f"{args.fs_in} Hz does not divide evenly into {args.fs_out} Hz")

    if factor > 1:
        y = decimate_block(y, factor)
        idx = np.arange(0, len(ts), factor)[: len(y)]
        ts, phase, label, eco = ts[idx], phase[idx], label[idx], eco[idx]
        y = y[: len(idx)]
        print(f"decimated {factor}x with an anti-alias filter -> {args.fs_out} Hz, {len(y)} samples")

    out = pd.DataFrame({
        "timestamp": ts,
        "phase": phase,
        "label": label,
        "ecological_modality": eco,
    })
    for i, name in enumerate(CHANNELS):
        out[name] = y[:, i]

    stamp = os.path.basename(os.path.normpath(args.session_dir))
    out_path = os.path.join(args.session_dir, f"eeg_data_{stamp}.csv")
    out.to_csv(out_path, index=False)
    print(f"wrote {out_path}  ({len(out)} rows)")


if __name__ == "__main__":
    main()
