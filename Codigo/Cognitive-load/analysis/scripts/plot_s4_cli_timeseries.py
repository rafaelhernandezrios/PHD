#!/usr/bin/env python3
"""
Publication figure: Session S4 (heberto) θFz/αPz CLI vs concatenated valid windows.
Reuses preprocessing and CLI from step4_cognitive_load_cleaned (Welch nperseg=250,
noverlap=125, Hann window; theta 4–7 Hz at Fz, alpha 8–12 Hz at Pz — IOP manuscript).
"""

from __future__ import annotations

import argparse
import contextlib
import io
import os
import sys
from pathlib import Path
from typing import Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# Pipeline (same CLI definition as IOP manuscript / step4)
_PIPELINE = Path(__file__).resolve().parent.parent / "pipeline"
sys.path.insert(0, str(_PIPELINE))
from step4_cognitive_load_cleaned import analyze_subject_cleaned  # noqa: E402

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DEFAULT_EEG = BASE_DIR / "data" / "Data-Experimento-Rafa" / "data_heberto" / "eeg_data_20260123_114745.csv"
OUT_DIR = BASE_DIR / "output" / "figures"


def _mm_to_in(mm: float) -> float:
    return mm / 25.4


def _filter_outliers(values: np.ndarray, method: str, threshold: float) -> Tuple[np.ndarray, np.ndarray]:
    """Return filtered values and a boolean mask of kept samples."""
    if method == "none":
        mask = np.ones(len(values), dtype=bool)
        return values, mask

    if len(values) == 0:
        mask = np.zeros(0, dtype=bool)
        return values, mask

    if method == "iqr":
        q1 = np.percentile(values, 25)
        q3 = np.percentile(values, 75)
        iqr = q3 - q1
        if iqr <= 0:
            mask = np.ones(len(values), dtype=bool)
            return values, mask
        lower = q1 - threshold * iqr
        upper = q3 + threshold * iqr
        mask = (values >= lower) & (values <= upper)
        return values[mask], mask

    # method == "mad" (robust z-score)
    median = np.median(values)
    mad = np.median(np.abs(values - median))
    if mad <= 0:
        mask = np.ones(len(values), dtype=bool)
        return values, mask

    robust_z = 0.6745 * (values - median) / mad
    mask = np.abs(robust_z) <= threshold
    return values[mask], mask


def plot_figure(
    low: np.ndarray,
    high: np.ndarray,
    out_png: Path,
    out_pdf: Path,
    width_mm: float = 180.0,
    height_mm: float = 58.0,
) -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif", "Bitstream Vera Serif"],
            "font.size": 10,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "axes.linewidth": 0.9,
            "lines.linewidth": 1.0,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "figure.dpi": 300,
            "savefig.dpi": 300,
        }
    )

    n_low, n_high = len(low), len(high)
    x_low = np.arange(1, n_low + 1, dtype=float)
    x_high = np.arange(n_low + 1, n_low + n_high + 1, dtype=float)
    x_all = np.concatenate([x_low, x_high])

    m_low, s_low = float(np.mean(low)), float(np.std(low, ddof=1))
    m_high, s_high = float(np.mean(high)), float(np.std(high, ddof=1))

    blue, red = "#1f4e79", "#b71c1c"
    blue_fill, red_fill = "#8fa8c8", "#e8a5a0"

    w_in, h_in = _mm_to_in(width_mm), _mm_to_in(height_mm)
    fig, ax = plt.subplots(figsize=(w_in, h_in), layout="constrained")

    ax.fill_between(
        x_low,
        m_low - s_low,
        m_low + s_low,
        color=blue_fill,
        alpha=0.45,
        linewidth=0,
        zorder=1,
    )
    ax.fill_between(
        x_high,
        m_high - s_high,
        m_high + s_high,
        color=red_fill,
        alpha=0.45,
        linewidth=0,
        zorder=1,
    )

    ax.hlines(m_low, x_low[0], x_low[-1], colors=blue, linestyles="-", linewidth=1.0, zorder=2)
    ax.hlines(m_high, x_high[0], x_high[-1], colors=red, linestyles="-", linewidth=1.0, zorder=2)

    ax.plot(x_low, low, color=blue, marker="o", markersize=2.2, linestyle="-", linewidth=0.85, zorder=3, label="Low load")
    ax.plot(x_high, high, color=red, marker="o", markersize=2.2, linestyle="-", linewidth=0.85, zorder=3, label="High load")

    ax.axvline(n_low + 0.5, color="0.35", linestyle="--", linewidth=1.0, zorder=4)

    ax.set_xlabel("Concatenated valid window index")
    ax.set_ylabel(r"CLI ($\theta_{Fz}/\alpha_{Pz}$)")
    ax.set_xlim(0.5, n_low + n_high + 0.5)
    ax.margins(x=0)

    y_min = float(
        np.min(
            [
                np.min(low),
                np.min(high),
                m_low - s_low,
                m_high - s_high,
            ]
        )
    )
    y_max = float(
        np.max(
            [
                np.max(low),
                np.max(high),
                m_low + s_low,
                m_high + s_high,
            ]
        )
    )
    pad = 0.06 * (y_max - y_min if y_max > y_min else 1.0)
    ax.set_ylim(y_min - pad, y_max + pad)

    ax.grid(axis="y", linestyle="-", alpha=0.12, linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(loc="upper left", frameon=True, framealpha=0.95, edgecolor="0.85")

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, format="png", dpi=300, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out_pdf, format="pdf", dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="S4 CLI time series (controlled paradigm, session heberto).")
    p.add_argument(
        "--eeg-csv",
        type=Path,
        default=DEFAULT_EEG,
        help="Preprocessed EEG CSV (labels: low_cognitive_load / high_cognitive_load).",
    )
    p.add_argument("--out-dir", type=Path, default=OUT_DIR, help="Output directory for PNG/PDF.")
    p.add_argument("--width-mm", type=float, default=180.0, help="Figure width (IOP double column ~180 mm).")
    p.add_argument("--height-mm", type=float, default=58.0)
    p.add_argument(
        "--outlier-method",
        choices=["none", "mad", "iqr"],
        default="none",
        help="Outlier filtering method applied independently to low/high CLI series.",
    )
    p.add_argument(
        "--outlier-threshold",
        type=float,
        default=3.5,
        help="Threshold for selected outlier method (MAD robust-z or IQR multiplier).",
    )
    p.add_argument(
        "--output-prefix",
        type=str,
        default="s4_cli_timeseries",
        help="Output filename prefix (without extension).",
    )
    args = p.parse_args()

    if not args.eeg_csv.is_file():
        raise SystemExit(f"Missing EEG file: {args.eeg_csv}")

    tmp = os.environ.get("TMPDIR", "/tmp")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        result = analyze_subject_cleaned(str(args.eeg_csv), tmp)

    low = np.asarray(result["phases"]["low_cognitive_load"]["ratios_after"], dtype=float)
    high = np.asarray(result["phases"]["high_cognitive_load"]["ratios_after"], dtype=float)

    low_filtered, low_mask = _filter_outliers(low, args.outlier_method, args.outlier_threshold)
    high_filtered, high_mask = _filter_outliers(high, args.outlier_method, args.outlier_threshold)

    low_removed = int((~low_mask).sum())
    high_removed = int((~high_mask).sum())

    if len(low) != 59 or len(high) != 64:
        print(f"Warning: expected 59/64 windows (S4); got {len(low)}/{len(high)}.", file=sys.stderr)

    if len(low_filtered) == 0 or len(high_filtered) == 0:
        raise SystemExit(
            "Outlier filtering removed all values in at least one phase. "
            "Try a larger --outlier-threshold or --outlier-method none."
        )

    out_png = args.out_dir / f"{args.output_prefix}.png"
    out_pdf = args.out_dir / f"{args.output_prefix}.pdf"
    plot_figure(low_filtered, high_filtered, out_png, out_pdf, width_mm=args.width_mm, height_mm=args.height_mm)
    print(
        f"Outlier filtering: method={args.outlier_method}, threshold={args.outlier_threshold}, "
        f"removed low={low_removed}/{len(low)} high={high_removed}/{len(high)}"
    )
    print(f"Wrote {out_png}")
    print(f"Wrote {out_pdf}")


if __name__ == "__main__":
    main()
