"""
Leave-one-subject-out accuracy (low vs high cognitive load) for subjects that
meet the spectral hypothesis (cumplen incluibles). Uses the same window
features as AI-Cognitive (eeg_make_window_features).

Outputs PNG + CSV under Cognitive-load/output/analysis_output/
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# --- load feature extraction from AI-Cognitive (same FS / windowing) ---
PHD_ROOT = Path(__file__).resolve().parents[2]
WMF_PATH = PHD_ROOT / "AI-Cognitive" / "scripts" / "eeg_make_window_features.py"
_spec = importlib.util.spec_from_file_location("eeg_make_window_features", WMF_PATH)
wmf = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(wmf)

FS = wmf.FS
WINDOW_SIZE = wmf.WINDOW_SIZE
STEP_SIZE = wmf.STEP_SIZE
compute_features_for_window = wmf.compute_features_for_window

BASE = Path(__file__).resolve().parents[1]
OUTPUT_DIR = BASE.parent / "output" / "analysis_output"
DATA_RAFA = BASE.parent / "data" / "Data-Experimento-Rafa"
CUMPLEAN_CSV = OUTPUT_DIR / "cumplen_incluibles_summary_fz_pz.csv"


def _subject_to_folder_name(subject: str) -> str:
    s = subject.strip()
    return f"data_{s}"


def _list_eeg_csvs(folder: Path) -> list[Path]:
    return sorted(folder.glob("eeg_data_*.csv"))


def windows_from_rafa_csv(
    csv_path: Path,
    subject_id: str,
) -> list[dict]:
    """One dict per window with feature keys + subject_id + y ('low' / 'high')."""
    df = pd.read_csv(csv_path)
    if "label" not in df.columns:
        return []
    ch_cols = sorted(
        [c for c in df.columns if c.startswith("channel_")],
        key=lambda x: int(x.replace("channel_", "")),
    )
    if len(ch_cols) < 2:
        return []

    df = df.sort_values("timestamp" if "timestamp" in df.columns else df.columns[0])
    df = df[df["label"].isin(["low_cognitive_load", "high_cognitive_load"])].copy()
    if df.empty:
        return []

    value_cols = [f"v{i}" for i in range(len(ch_cols))]
    for i, c in enumerate(ch_cols):
        df[value_cols[i]] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    df["run"] = (df["label"] != df["label"].shift()).cumsum()
    file_name = csv_path.name
    out: list[dict] = []

    for _, g in df.groupby("run", sort=True):
        g = g.reset_index(drop=True)
        lab = str(g.loc[0, "label"])
        y = "low" if lab == "low_cognitive_load" else "high"
        data = g[value_cols].to_numpy(dtype=np.float64)
        n = len(g)
        start = 0
        while start + WINDOW_SIZE <= n:
            end = start + WINDOW_SIZE
            segment = data[start:end, :]
            feats = compute_features_for_window(segment, value_cols)
            feats["subject_id"] = subject_id
            feats["file_name"] = file_name
            feats["y"] = y
            out.append(feats)
            start += STEP_SIZE
    return out


def load_cumplen_subjects() -> list[str]:
    if not CUMPLEAN_CSV.is_file():
        raise FileNotFoundError(f"Missing {CUMPLEAN_CSV}")
    s = pd.read_csv(CUMPLEAN_CSV)["subject"].astype(str).tolist()
    return sorted(set(s), key=lambda x: x.lower())


def build_all_windows(subjects: list[str]) -> pd.DataFrame:
    rows: list[dict] = []
    for sub in subjects:
        folder = DATA_RAFA / _subject_to_folder_name(sub)
        if not folder.is_dir():
            print(f"[skip] no folder {folder}")
            continue
        for csv_p in _list_eeg_csvs(folder):
            rows.extend(windows_from_rafa_csv(csv_p, subject_id=sub))
    if not rows:
        raise RuntimeError("No windows built. Check DATA_RAFA paths and CSV labels.")
    return pd.DataFrame(rows)


def leave_one_subject_out_accuracy(df: pd.DataFrame) -> pd.DataFrame:
    """RandomForest, balanced classes; LOSO per subject present in df."""
    feature_cols = [
        c
        for c in df.columns
        if c not in {"subject_id", "file_name", "y"}
        and pd.api.types.is_numeric_dtype(df[c])
    ]
    subjects = sorted(df["subject_id"].unique())
    results = []
    for test_sub in subjects:
        train_mask = df["subject_id"] != test_sub
        test_mask = df["subject_id"] == test_sub
        X_train = df.loc[train_mask, feature_cols].to_numpy(dtype=np.float64)
        y_train = df.loc[train_mask, "y"].to_numpy()
        X_test = df.loc[test_mask, feature_cols].to_numpy(dtype=np.float64)
        y_test = df.loc[test_mask, "y"].to_numpy()
        X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
        X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)
        X_train = np.clip(X_train, -1e10, 1e10)
        X_test = np.clip(X_test, -1e10, 1e10)

        if len(np.unique(y_train)) < 2 or len(y_test) == 0:
            results.append(
                {
                    "subject": test_sub,
                    "accuracy": np.nan,
                    "n_test_windows": int(len(y_test)),
                    "note": "missing class or no test windows",
                }
            )
            continue

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_train)
        X_te = scaler.transform(X_test)

        clf = RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced",
        )
        clf.fit(X_tr, y_train)
        pred = clf.predict(X_te)
        acc = float(accuracy_score(y_test, pred))
        results.append(
            {
                "subject": test_sub,
                "accuracy": acc,
                "n_test_windows": int(len(y_test)),
                "note": "",
            }
        )
    return pd.DataFrame(results)


def plot_bar(df_res: pd.DataFrame, out_png: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    subs = df_res["subject"].tolist()
    acc = df_res["accuracy"].to_numpy()
    colors = ["#2ecc71" if not np.isnan(a) else "#95a5a6" for a in acc]
    x = np.arange(len(subs))
    bars = ax.bar(x, np.nan_to_num(acc, nan=0.0), color=colors, edgecolor="#333", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(subs, rotation=25, ha="right")
    ax.set_ylim(0, 1.05)
    ax.axhline(0.5, color="#7f8c8d", linestyle="--", linewidth=1, label="chance (2 classes)")
    ax.set_ylabel("Accuracy (LOSO)")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=9)
    # annotate n_test
    for i, (_, row) in enumerate(df_res.iterrows()):
        n = int(row["n_test_windows"])
        a = row["accuracy"]
        label = f"n={n}" if np.isnan(a) else f"{a:.2f}\nn={n}"
        ax.text(
            i,
            (0.02 if np.isnan(a) else min(acc[i] + 0.04, 1.0)),
            label,
            ha="center",
            va="bottom",
            fontsize=8,
        )
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    subjects = load_cumplen_subjects()
    print("Cumplen subjects:", subjects)
    df = build_all_windows(subjects)
    print(f"Total windows: {len(df)}; columns: {df.shape[1]}")
    counts = df.groupby(["subject_id", "y"]).size().unstack(fill_value=0)
    print("Windows per subject / class:\n", counts)

    df_res = leave_one_subject_out_accuracy(df)
    out_csv = OUTPUT_DIR / "loso_accuracy_cumplen_subjects.csv"
    df_res.to_csv(out_csv, index=False)
    print(df_res)
    print(f"Wrote {out_csv}")

    title = (
        "LOSO accuracy: low vs high cognitive load\n"
        "(RandomForest, subjects with spectral hypothesis High > Low)"
    )
    out_png = OUTPUT_DIR / "loso_accuracy_cumplen_subjects.png"
    plot_bar(df_res, out_png, title)
    print(f"Wrote {out_png}")


if __name__ == "__main__":
    main()
