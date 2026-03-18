from pathlib import Path
from typing import List, Tuple

import csv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = PROJECT_ROOT / "raw_data"


def parse_line(line: str) -> Tuple[List[float], str]:
    """
    Parse one raw line from the .txt files.

    - Removes 'Lxxx:' prefix if present.
    - Splits by comma.
    - Treats the last value as timestamp if it contains '-' or ':'.
    - Returns (numeric_values, timestamp_string_or_empty).
    """
    line = line.strip()
    if not line:
        return [], ""

    # remove optional "L123:" prefix
    if ":" in line and line.split(":", 1)[0].startswith("L"):
        line = line.split(":", 1)[1].strip()

    parts = [p.strip() for p in line.split(",")]
    if not parts:
        return [], ""

    # detect timestamp-like last field
    timestamp = ""
    if any(c in parts[-1] for c in ("-", ":", " ")):
        timestamp = parts[-1]
        parts = parts[:-1]

    numeric_values: List[float] = []
    for p in parts:
        if not p:
            numeric_values.append(0.0)
            continue
        try:
            numeric_values.append(float(p))
        except ValueError:
            # if parsing fails, set to 0.0
            numeric_values.append(0.0)

    return numeric_values, timestamp


def map_condition(stem_prefix: str) -> Tuple[str, str]:
    """
    Map filename prefix (natural, lowlevel, midlevel, highlevel)
    to (condition_4, load_3).

    condition_4: normal / low / mid / high  (as in the paper)
    load_3:     normal / low / high (coarse mapping for your project)
    """
    prefix = stem_prefix.lower()

    # 4-level label from the paper
    if prefix == "natural":
        condition_4 = "normal"
    elif prefix == "lowlevel":
        condition_4 = "low"
    elif prefix == "midlevel":
        condition_4 = "mid"
    elif prefix == "highlevel":
        condition_4 = "high"
    else:
        return "unknown", "unknown"

    # 3-level mapping: low / normal / high (carga cognitiva)
    # - lowlevel  -> low   (carga baja)
    # - natural   -> normal (línea base)
    # - midlevel + highlevel -> high (carga media/alta)
    # Así "low" es más separable (lowlevel vs natural) que antes (midlevel en medio).
    if condition_4 == "low":   # lowlevel
        load_3 = "low"
    elif condition_4 == "normal":  # natural
        load_3 = "normal"
    else:  # mid + high -> high
        load_3 = "high"

    return condition_4, load_3


def collect_rows() -> Tuple[list[list], int]:
    """
    Traverse Arithmetic_Data and Stroop_Data, read all .txt files
    and collect rows with metadata.

    Returns (rows, max_num_values) where:
    - rows is a list of [v0..vK-1, timestamp, subject_id, task_type,
                         condition_4, load_3, file_name]
    - max_num_values is the maximum length K seen across all lines.
    """
    rows: list[list] = []
    max_num_values = 0

    for task_dir in ("Arithmetic_Data", "Stroop_Data"):
        task_path = RAW_ROOT / task_dir
        if not task_path.exists():
            continue

        task_type = task_dir.replace("_Data", "").lower()  # "arithmetic" / "stroop"

        for txt_path in sorted(task_path.glob("*.txt")):
            file_name = txt_path.name
            stem = txt_path.stem  # e.g. "lowlevel-1"

            if "-" in stem:
                prefix, idx_str = stem.split("-", 1)
            else:
                prefix, idx_str = stem, "0"

            condition_4, load_3 = map_condition(prefix)

            # simple subject id from suffix number (1..15)
            try:
                subject_id = int(idx_str)
            except ValueError:
                subject_id = -1

            with txt_path.open("r", encoding="utf-8") as f:
                for line in f:
                    values, timestamp = parse_line(line)
                    if not values:
                        continue

                    if len(values) > max_num_values:
                        max_num_values = len(values)

                    row = (
                        values
                        + [timestamp, subject_id, task_type, condition_4, load_3, file_name]
                    )
                    rows.append(row)

    return rows, max_num_values


def build_header(num_value_cols: int) -> list[str]:
    """
    Build generic header names for value columns plus metadata.

    Value columns are named v0..v{N-1}. We do not assume
    a specific mapping to Fp1..C4 here to stay close to raw format.
    """
    value_cols = [f"v{i}" for i in range(num_value_cols)]
    meta_cols = [
        "timestamp",
        "subject_id",
        "task_type",
        "condition_4",
        "load_3",
        "file_name",
    ]
    return value_cols + meta_cols


def main() -> None:
    rows, max_num_values = collect_rows()

    if not rows:
        print("No rows found under", RAW_ROOT)
        return

    header = build_header(max_num_values)

    # pad rows with fewer numeric values so all have the same length
    expected_len = max_num_values + 6  # timestamp + 5 metadata columns
    for r in rows:
        if len(r) < expected_len:
            # pad numeric part with zeros until v0..v{N-1} length is max_num_values
            missing = expected_len - len(r)
            r[:0] = [0.0] * missing  # prepend zeros if needed

    out_dir = PROJECT_ROOT / "csv"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "eeg_all_samples.csv"

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    print(
        f"Wrote {len(rows)} rows with {len(header)} columns to {out_csv} "
        f"(value columns: {max_num_values})"
    )


if __name__ == "__main__":
    main()

