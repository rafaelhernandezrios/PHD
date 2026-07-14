#!/usr/bin/env python3
"""Copy and organize EEG CSVs used in the Wiley cognitive-load paper."""

from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parents[3] / "Wiley-Rafa"

CONTROLLED = {
    "S01": BASE / "data/Data-Experimento-Rafa/data_1/eeg_data_20260122_120044.csv",
    "S02": BASE / "data/Data-Experimento-Rafa/data_2/eeg_data_20260122_123839.csv",
    "S03": BASE / "data/Data-Experimento-Rafa/data_3/eeg_data_20260122_131125.csv",
    "S04": BASE / "data/Data-Experimento-Rafa/data_4/eeg_data_20260122_134321.csv",
    "S05": BASE / "electron/data_clau2/eeg_data_20260417_182251.csv",
    "S06": BASE / "data/Data-Experimento-Rafa/data_heberto/eeg_data_20260123_114745.csv",
    "S07": BASE / "electron/data_jonhy/eeg_data_20260417_160903.csv",
    "S08": BASE / "electron/data_michelle/eeg_data_20260414_182434.csv",
}

ECOLOGICAL = {
    "E07_keyboard": (
        BASE / "electron/data_jonhy/eeg_data_20260417_160903.csv",
        "eco_keyboard",
    ),
    "E07_haptic": (
        BASE / "electron/data_jonhy/eeg_data_20260417_160903.csv",
        "eco_haptic",
    ),
    "E08_keyboard": (
        BASE / "electron/data_michelle/eeg_data_20260414_182434.csv",
        "eco_keyboard",
    ),
    "E08_haptic": (
        BASE / "electron/data_michelle/eeg_data_20260414_182434.csv",
        "eco_haptic",
    ),
}

BASELINE_LABELS = {"baseline_eyes_open", "baseline_eyes_closed"}


def copy_controlled(session_id: str, src: Path, dest_dir: Path) -> dict:
    dest = dest_dir / session_id / "eeg_session.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    df = pd.read_csv(src, usecols=["label"])
    phases = sorted(df["label"].dropna().unique().tolist())
    return {
        "paradigm": "controlled",
        "session_id": session_id,
        "source_file": str(src.relative_to(BASE)),
        "destination": str(dest.relative_to(OUT)),
        "n_rows": len(df),
        "phases": ";".join(phases),
        "ecological_modality": "",
    }


def extract_ecological(session_id: str, src: Path, modality: str, dest_dir: Path) -> dict:
    df = pd.read_csv(src)
    eco_label = f"ecological_paradigm mode={modality}"
    mask = df["label"].isin(BASELINE_LABELS) | (
        (df["label"] == eco_label) & (df.get("ecological_modality", modality) == modality)
    )
    if "ecological_modality" in df.columns:
        mask = df["label"].isin(BASELINE_LABELS) | (
            (df["label"] == eco_label) & (df["ecological_modality"] == modality)
        )
    subset = df.loc[mask].copy()
    dest = dest_dir / session_id / "eeg_session.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    subset.to_csv(dest, index=False)
    phases = sorted(subset["label"].dropna().unique().tolist())
    return {
        "paradigm": "ecological",
        "session_id": session_id,
        "source_file": str(src.relative_to(BASE)),
        "destination": str(dest.relative_to(OUT)),
        "n_rows": len(subset),
        "phases": ";".join(phases),
        "ecological_modality": modality,
    }


def write_readme() -> None:
    readme = """# Wiley-Rafa — EEG dataset

Raw EEG recordings used in the cognitive workload study (controlled laboratory protocol and ecological exergaming task). Participant identifiers are anonymized.

## Structure

```
controlled/     Primary analysis (S01–S08), fs ≈ 50 Hz
ecological/     Exergaming segments with baseline (E07_*, E08_*)
sessions_manifest.csv
```

## CSV format

Columns (LSL export from the AURA 8-channel headset):

| Column | Description |
|--------|-------------|
| `timestamp` | LSL stream timestamp (seconds) |
| `phase` | Internal acquisition phase |
| `label` | Canonical phase label (`baseline_eyes_open`, `baseline_eyes_closed`, `low_cognitive_load`, `high_cognitive_load`, `ecological_paradigm mode=...`) |
| `ecological_modality` | Present in ecological sessions: `eco_keyboard`, `eco_haptic`, or `eco_gamepad` |
| `channel_0` … `channel_7` | Fp1, Fp2, F3, Fz, F4, P3, Pz, P4 (µV) |

Nominal sampling rate: 250 Hz with 5× subsampling → ~50 Hz effective rate in retained sessions.

## Controlled paradigm (S01–S08)

Eight sessions included in the primary Low–High workload analysis after the sampling-rate criterion (effective fs ≥ 45 Hz). S02 has no valid High-load segment in the published window counts.

## Ecological paradigm

Four modality-specific exports are provided (keyboard and haptic for sessions linked to S07 and S08). Each file retains baseline eyes-open/closed segments from the same recording plus the corresponding exergaming segment.

The paper reports eight ecological sessions (E01–E08); six earlier sessions contributed summary statistics in the published figure but are not available in this LSL CSV export format.

## Citation

If you use this dataset, please cite the corresponding Wiley/IOP manuscript and link to this repository: https://github.com/rafaelhernandezrios/Wiley-Rafa
"""
    (OUT / "README.md").write_text(readme, encoding="utf-8")


def main() -> None:
    if not OUT.exists():
        raise SystemExit(f"Output repo not found: {OUT}")

    controlled_dir = OUT / "controlled"
    ecological_dir = OUT / "ecological"
    controlled_dir.mkdir(parents=True, exist_ok=True)
    ecological_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for sid, src in CONTROLLED.items():
        if not src.is_file():
            raise FileNotFoundError(src)
        rows.append(copy_controlled(sid, src, controlled_dir))

    for sid, (src, modality) in ECOLOGICAL.items():
        if not src.is_file():
            raise FileNotFoundError(src)
        rows.append(extract_ecological(sid, src, modality, ecological_dir))

    manifest = pd.DataFrame(rows)
    manifest.to_csv(OUT / "sessions_manifest.csv", index=False)
    write_readme()
    print(f"Dataset written to {OUT}")
    print(manifest.to_string(index=False))


if __name__ == "__main__":
    main()
