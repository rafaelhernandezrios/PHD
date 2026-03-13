from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.signal import welch


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# sampling rate from the paper
FS = 250.0

# window parameters in samples
WINDOW_SIZE = int(2 * FS)  # 2-second windows
STEP_SIZE = int(1 * FS)  # 1-second step (50% overlap)


def load_eeg_samples(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    required_cols = {"file_name", "condition_4", "load_3", "task_type", "subject_id"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"CSV must contain columns: {required_cols}")

    # Keep only rows with known labels
    df = df[df["condition_4"] != "unknown"].copy()
    df = df.reset_index(drop=True)

    return df


def bandpower(
    data: np.ndarray,
    sf: float,
    band: Tuple[float, float],
    window_sec: Optional[float] = None,
) -> float:
    """
    Compute band power using Welch's method.
    """
    low, high = band
    if window_sec is None:
        window_sec = (len(data) / sf)

    nperseg = int(window_sec * sf)
    if nperseg > len(data):
        nperseg = len(data)

    freqs, psd = welch(data, sf, nperseg=nperseg)

    idx_band = np.logical_and(freqs >= low, freqs <= high)
    return np.trapz(psd[idx_band], freqs[idx_band])


def compute_features_for_window(
    segment: np.ndarray,
    value_cols: List[str],
) -> Dict[str, float]:
    """
    Compute time- and frequency-domain features for one multi-channel window.

    `segment` has shape (window_len, n_channels) and columns in `value_cols`.
    """
    features: dict = {}

    # time-domain features per channel
    for i, col in enumerate(value_cols):
        x = segment[:, i]
        features[f"{col}_mean"] = float(np.mean(x))
        features[f"{col}_std"] = float(np.std(x))
        features[f"{col}_min"] = float(np.min(x))
        features[f"{col}_max"] = float(np.max(x))

    # frequency-domain features per channel: band powers + simple ratios
    bands: Dict[str, Tuple[float, float]] = {
        "delta": (1.0, 4.0),
        "theta": (4.0, 8.0),
        "alpha": (8.0, 13.0),
        "beta": (13.0, 30.0),
        "gamma": (30.0, 40.0),
    }

    for i, col in enumerate(value_cols):
        x = segment[:, i]
        bp = {
            name: bandpower(x, FS, band)
            for name, band in bands.items()
        }

        for name, power in bp.items():
            features[f"{col}_bp_{name}"] = float(power)

        alpha_p = bp["alpha"] if bp["alpha"] > 0 else 1e-9
        features[f"{col}_theta_alpha_ratio"] = float(bp["theta"] / alpha_p)
        features[f"{col}_beta_alpha_ratio"] = float(bp["beta"] / alpha_p)

    return features


def build_window_features(df: pd.DataFrame) -> pd.DataFrame:
    # value columns: start with "v" and are numeric
    value_cols = [
        c
        for c in df.columns
        if c.startswith("v") and pd.api.types.is_numeric_dtype(df[c])
    ]

    if not value_cols:
        raise ValueError("No numeric value columns starting with 'v' were found.")

    all_features: list[dict] = []

    # group by file/subject/condition to preserve temporal structure
    group_cols = ["file_name", "subject_id", "task_type", "condition_4", "load_3"]

    for _, g in df.groupby(group_cols, sort=False):
        g = g.reset_index(drop=True)

        data = g[value_cols].to_numpy()
        n = len(g)

        start = 0
        while start + WINDOW_SIZE <= n:
            end = start + WINDOW_SIZE
            segment = data[start:end, :]

            feats = compute_features_for_window(segment, value_cols)

            # metadata for this window
            feats["file_name"] = g.loc[0, "file_name"]
            feats["subject_id"] = g.loc[0, "subject_id"]
            feats["task_type"] = g.loc[0, "task_type"]
            feats["condition_4"] = g.loc[0, "condition_4"]
            feats["load_3"] = g.loc[0, "load_3"]
            feats["window_start_idx"] = int(start)
            feats["window_end_idx"] = int(end)

            all_features.append(feats)

            start += STEP_SIZE

    if not all_features:
        raise RuntimeError("No window features were generated. Check window size and data length.")

    features_df = pd.DataFrame(all_features)
    return features_df


def main() -> None:
    eeg_csv = PROJECT_ROOT / "csv" / "eeg_all_samples.csv"
    if not eeg_csv.exists():
        raise FileNotFoundError(
            f"{eeg_csv} not found. Run 'eeg_convert_raw_to_clean.py' first."
        )

    df = load_eeg_samples(eeg_csv)
    features_df = build_window_features(df)

    out_dir = PROJECT_ROOT / "csv"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "eeg_window_features.csv"

    features_df.to_csv(out_csv, index=False)
    print(
        f"Wrote {len(features_df)} window-level rows with "
        f"{features_df.shape[1]} columns to {out_csv}"
    )


if __name__ == "__main__":
    main()

