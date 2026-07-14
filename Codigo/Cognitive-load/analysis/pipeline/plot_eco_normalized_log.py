"""
Dot plot (scatter) of normalized theta/alpha ratio for ecological modalities.
Computes task/baseline from subject CSVs and includes only quality-valid points.
"""

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_PIPELINE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_PIPELINE_DIR))
from step4_cognitive_load_cleaned import (  # noqa: E402
    AMPLITUDE_THRESHOLD,
    FZ_CHANNEL,
    IQR_MULTIPLIER,
    PZ_CHANNEL,
    SAMPLE_RATE,
    Z_SCORE_THRESHOLD,
    apply_car,
    calculate_cognitive_load_by_phase,
    detect_artifacts_combined,
    preprocess_signal,
    remove_artifacts_interpolation,
)

BASE_DIR = Path(__file__).resolve().parents[2]
OUTPUT_PATH = BASE_DIR / "output" / "analysis_output" / "eco_normalized_ratio_log.png"

DATASETS = [
    ("michelle", BASE_DIR / "data" / "Data-Experimento-Rafa" / "data_michelle" / "eeg_data_20260414_182434.csv"),
    ("jonhy", BASE_DIR / "electron" / "data_jonhy" / "eeg_data_20260417_160903.csv"),
]
ECO_PHASES = [
    ("eco_keyboard", "Keyboard"),
    ("eco_haptic", "Haptic"),
]

# Keep legacy ecological sessions already used in this figure.
LEGACY_POINTS = [
    ("S1", 7.85),
    ("S2", 1.87),
    ("S3", 2.51),
    ("S4", 3.14),
    ("S5", 4.08),
    ("S6", 2.14),
    ("S7", 0.71),
    ("S8", 0.33),
]

# Quality gates for "si el analisis lo amerita"
MIN_WINDOWS = 2
MAX_ARTIFACT_PCT = 50.0

# IEEE single-column width (approx 3.5 in), height to taste
FIG_W = 5.2
FIG_H = 2.8
FONT_SIZE = 8
TICK_SIZE = 8
ANNOT_SIZE = 7


def _compute_phase_ratio(df_phase: pd.DataFrame, sample_rate: float) -> dict:
    if len(df_phase) == 0:
        return {
            "mean_ratio_after": np.nan,
            "n_windows_after": 0,
            "fz_artifact_pct": np.nan,
            "pz_artifact_pct": np.nan,
        }

    phase_data = df_phase.copy()
    if "timestamp" in phase_data.columns:
        phase_data = phase_data.sort_values("timestamp")

    eeg_matrix = np.zeros((len(phase_data), 8))
    for ch in range(8):
        eeg_matrix[:, ch] = phase_data[f"channel_{ch}"].values

    use_default_for_filtering = sample_rate < 10
    eeg_filtered = np.zeros_like(eeg_matrix)
    for ch in range(8):
        eeg_filtered[:, ch] = preprocess_signal(
            eeg_matrix[:, ch],
            sample_rate,
            use_default_sr=use_default_for_filtering,
        )

    eeg_car = apply_car(eeg_filtered)
    fz_signal = eeg_car[:, FZ_CHANNEL].copy()
    pz_signal = eeg_car[:, PZ_CHANNEL].copy()

    fz_artifacts, fz_n_artifacts = detect_artifacts_combined(
        fz_signal,
        z_threshold=Z_SCORE_THRESHOLD,
        iqr_multiplier=IQR_MULTIPLIER,
        amp_threshold=AMPLITUDE_THRESHOLD,
    )
    pz_artifacts, pz_n_artifacts = detect_artifacts_combined(
        pz_signal,
        z_threshold=Z_SCORE_THRESHOLD,
        iqr_multiplier=IQR_MULTIPLIER,
        amp_threshold=AMPLITUDE_THRESHOLD,
    )

    fz_cleaned = remove_artifacts_interpolation(fz_signal, fz_artifacts)
    pz_cleaned = remove_artifacts_interpolation(pz_signal, pz_artifacts)

    ratios_after, _, _ = calculate_cognitive_load_by_phase(
        fz_cleaned,
        pz_cleaned,
        SAMPLE_RATE,
    )

    return {
        "mean_ratio_after": float(np.mean(ratios_after)) if ratios_after else np.nan,
        "n_windows_after": len(ratios_after),
        "fz_artifact_pct": float(100 * fz_n_artifacts / len(fz_signal)) if len(fz_signal) else np.nan,
        "pz_artifact_pct": float(100 * pz_n_artifacts / len(pz_signal)) if len(pz_signal) else np.nan,
    }


def _compute_subject_points(subject_name: str, csv_path: Path) -> list:
    if not csv_path.is_file():
        print(f"[WARN] Missing CSV for {subject_name}: {csv_path}")
        return []

    df = pd.read_csv(csv_path)
    if "timestamp" in df.columns:
        ts = pd.to_numeric(df["timestamp"], errors="coerce")
        median_interval = ts.sort_values().diff().dropna().median()
        sample_rate = 1.0 / median_interval if median_interval and median_interval > 0 else SAMPLE_RATE
    else:
        sample_rate = SAMPLE_RATE

    baseline_candidates = [
        _compute_phase_ratio(df[df["label"] == "baseline_eyes_closed"], sample_rate),
        _compute_phase_ratio(df[df["label"] == "baseline_eyes_open"], sample_rate),
    ]
    baseline_ratio = np.nan
    for base in baseline_candidates:
        v = base["mean_ratio_after"]
        if np.isfinite(v) and v > 0:
            baseline_ratio = v
            break

    if not np.isfinite(baseline_ratio) or baseline_ratio <= 0:
        print(f"[WARN] {subject_name}: baseline invalido, omitido.")
        return []

    points = []
    for modality_key, modality_label in ECO_PHASES:
        phase_df = df[
            (df["label"] == f"ecological_paradigm mode={modality_key}")
            & (df["ecological_modality"] == modality_key)
        ]
        phase_metrics = _compute_phase_ratio(phase_df, sample_rate)

        ratio = phase_metrics["mean_ratio_after"]
        n_windows = phase_metrics["n_windows_after"]
        fz_pct = phase_metrics["fz_artifact_pct"]
        pz_pct = phase_metrics["pz_artifact_pct"]

        valid = (
            np.isfinite(ratio)
            and ratio > 0
            and n_windows >= MIN_WINDOWS
            and np.isfinite(fz_pct)
            and np.isfinite(pz_pct)
            and fz_pct <= MAX_ARTIFACT_PCT
            and pz_pct <= MAX_ARTIFACT_PCT
        )
        if not valid:
            print(
                f"[INFO] {subject_name}-{modality_key} omitido "
                f"(ratio={ratio:.3f}, windows={n_windows}, fz={fz_pct:.1f}%, pz={pz_pct:.1f}%)."
            )
            continue

        normalized = ratio / baseline_ratio
        points.append((f"{subject_name[:3].upper()}-{modality_label}", float(normalized)))
        print(
            f"[OK] {subject_name}-{modality_key}: "
            f"task={ratio:.3f}, baseline={baseline_ratio:.3f}, normalized={normalized:.3f}"
        )

    return points


def main() -> None:
    points = list(LEGACY_POINTS)
    for subject_name, csv_path in DATASETS:
        points.extend(_compute_subject_points(subject_name, csv_path))

    if not points:
        raise SystemExit("No valid ecological points available to plot.")

    labels = [p[0] for p in points]
    ratios = np.array([p[1] for p in points], dtype=float)

    plt.rcParams.update({"font.size": FONT_SIZE})
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    x = np.arange(len(labels))

    colors = ["#1976d2" if lb.startswith("S") else "#2e7d32" for lb in labels]
    ax.scatter(x, ratios, color=colors, s=30, zorder=3, edgecolors="#333", linewidths=0.8)

    for xi, yi in zip(x, ratios):
        ax.annotate(
            f"{yi:.2f}",
            (xi, yi),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            fontsize=ANNOT_SIZE,
            color="black",
        )

    ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=1, alpha=0.8, zorder=1)
    ax.set_yscale("log")
    ax.set_xlim(-0.5, len(labels) - 0.5)
    ax.set_ylim(0.2, max(15.0, float(np.nanmax(ratios)) * 1.2))

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_xlabel("Session / Subject-Modality", fontsize=FONT_SIZE)
    ax.set_ylabel(
        "Normalized Theta/Alpha (F region / P region)\n(task / baseline)",
        fontsize=FONT_SIZE,
    )
    ax.tick_params(axis="both", labelsize=TICK_SIZE)
    ax.yaxis.grid(True, linestyle="-", alpha=0.25)
    ax.set_axisbelow(True)

    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")
    for spine in ax.spines.values():
        spine.set_color("black")

    plt.tight_layout()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()

    print(f"Figure saved: {OUTPUT_PATH}")
    print(f"Points plotted: {labels}")


if __name__ == "__main__":
    # Avoid macOS AppKit backend issues in headless runs.
    os.environ.setdefault("MPLBACKEND", "Agg")
    main()
