from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.signal import butter, detrend, filtfilt, iirnotch, sosfiltfilt, welch
from scipy.stats import entropy


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# sampling rate from the paper
FS = 250.0

# window parameters in samples
WINDOW_SIZE = int(4 * FS)  # 4-second windows
STEP_SIZE = int(2 * FS)  # 2-second step (50% overlap)

# preprocessing: band-pass and notch (applied per channel before bandpower)
BANDPASS_LOW_HZ = 1.0
BANDPASS_HIGH_HZ = 40.0
BANDPASS_ORDER = 4
NOTCH_FREQS_HZ = (50.0, 60.0)  # notch 50 Hz and 60 Hz (zero-phase)
NOTCH_QUALITY = 30.0  # Q = f0/bw, higher = narrower notch


def load_eeg_samples(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    required_cols = {
        "file_name",
        "condition_4",
        "load_3",
        "load_2",
        "task_type",
        "subject_id",
    }
    if not required_cols.issubset(df.columns):
        raise ValueError(f"CSV must contain columns: {required_cols}")

    # Keep only rows with known labels
    df = df[df["condition_4"] != "unknown"].copy()
    df = df.reset_index(drop=True)

    return df


def _bandpass_filter(x: np.ndarray, low_hz: float, high_hz: float, fs: float, order: int) -> np.ndarray:
    """Zero-phase band-pass Butterworth (sos design for stability)."""
    nyq = fs / 2.0
    low = max(low_hz / nyq, 1e-6)
    high = min(high_hz / nyq, 1.0 - 1e-6)
    if low >= high:
        return x
    sos = butter(order, [low, high], btype="band", output="sos")
    return sosfiltfilt(sos, x.astype(np.float64), axis=0)


def _notch_filter(x: np.ndarray, f0_hz: float, fs: float, Q: float) -> np.ndarray:
    """Zero-phase notch at f0_hz (e.g. 50 or 60 Hz)."""
    w0 = f0_hz / (fs / 2.0)
    if w0 >= 1.0 or w0 <= 0.0:
        return x
    b, a = iirnotch(w0, Q)
    return filtfilt(b, a, x.astype(np.float64), axis=0)


def preprocess_for_bandpower(x: np.ndarray, fs: float = FS) -> np.ndarray:
    """
    Apply band-pass (1–40 Hz), detrend, then notch 50 Hz and 60 Hz.
    Used per channel before Welch bandpower.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return x
    # 1) Band-pass
    x = _bandpass_filter(x, BANDPASS_LOW_HZ, BANDPASS_HIGH_HZ, fs, BANDPASS_ORDER)
    # 2) Detrend (remove DC and linear drift in window)
    x = detrend(x, type="linear", axis=0)
    # 3) Notch 50 Hz and 60 Hz
    for f0 in NOTCH_FREQS_HZ:
        x = _notch_filter(x, f0, fs, NOTCH_QUALITY)
    return x


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
    trapz_fn = getattr(np, "trapezoid", np.trapz)  # trapezoid since NumPy 1.22
    return float(trapz_fn(psd[idx_band], freqs[idx_band]))


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
    bands: Dict[str, Tuple[float, float]] = {
        "delta": (1.0, 4.0),
        "theta": (4.0, 8.0),
        "alpha": (8.0, 13.0),
        "beta": (13.0, 30.0),
        "gamma": (30.0, 40.0),
    }

    for i, col in enumerate(value_cols):
        x_raw = segment[:, i]
        # Preprocess before bandpower: band-pass 1–40 Hz, detrend, notch 50/60 Hz
        x = preprocess_for_bandpower(x_raw, FS)

        # --- time-domain features on preprocessed signal ---
        features[f"{col}_mean"] = float(np.mean(x))
        features[f"{col}_std"] = float(np.std(x))
        features[f"{col}_min"] = float(np.min(x))
        features[f"{col}_max"] = float(np.max(x))
        # richer time-domain features
        features[f"{col}_rms"] = float(np.sqrt(np.mean(x ** 2)))
        features[f"{col}_abs_mean"] = float(np.mean(np.abs(x)))

        # --- frequency-domain: band powers + richer features ---
        bp = {name: bandpower(x, FS, band) for name, band in bands.items()}

        for name, power in bp.items():
            features[f"{col}_bp_{name}"] = float(power)

        total_power = float(sum(bp.values()) + 1e-12)
        features[f"{col}_bp_total"] = total_power

        # relative band powers
        for name, power in bp.items():
            features[f"{col}_rel_bp_{name}"] = float(power / total_power)

        # simple band ratios
        alpha_p = bp["alpha"] if bp["alpha"] > 0 else 1e-9
        features[f"{col}_theta_alpha_ratio"] = float(bp["theta"] / alpha_p)
        features[f"{col}_beta_alpha_ratio"] = float(bp["beta"] / alpha_p)

        # spectral entropy over 1–40 Hz
        freqs, psd = welch(x, FS, nperseg=min(len(x), int(4 * FS)))
        idx = np.logical_and(freqs >= 1.0, freqs <= 40.0)
        psd_band = psd[idx]
        psd_norm = psd_band / (psd_band.sum() + 1e-12)
        features[f"{col}_spec_entropy"] = float(entropy(psd_norm))

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
    group_cols = [
        "file_name",
        "subject_id",
        "task_type",
        "condition_4",
        "load_3",
        "load_2",
    ]

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
            feats["load_2"] = g.loc[0, "load_2"]
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

