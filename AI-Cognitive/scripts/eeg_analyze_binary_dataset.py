"""
Análisis paso a paso del dataset binario: clases **normal** vs **alta**.

Definición (columna load_2 en eeg_all_samples.csv):
  - normal = condición natural (línea base)
  - alta   = lowlevel, midlevel, highlevel (tarea con carga cognitiva)

Ejecutar DESPUÉS de:
  python scripts/eeg_convert_raw_to_clean.py
"""

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = PROJECT_ROOT / "csv" / "eeg_all_samples.csv"
FS = 250.0
WINDOW_SIZE = int(4 * FS)
STEP_SIZE = int(2 * FS)


def main() -> None:
    if not CSV_PATH.exists():
        raise FileNotFoundError(
            f"No existe {CSV_PATH}. Ejecuta primero eeg_convert_raw_to_clean.py"
        )

    df = pd.read_csv(CSV_PATH)

    print("=" * 60)
    print(" ANÁLISIS PASO A PASO — Dataset binario (normal vs alta)")
    print("=" * 60)

    if "load_2" not in df.columns:
        print(
            "\n[!] Falta la columna 'load_2'. Vuelve a generar el CSV con la versión\n"
            "    actual de eeg_convert_raw_to_clean.py\n"
        )
        return

    df_ok = df[df["load_2"].isin(["normal", "alta"])].copy()

    # --- Paso 1: tamaño general ---
    print("\n--- Paso 1: Tamaño del CSV ---")
    print(f"  Filas totales: {len(df):,}")
    print(f"  Columnas: {len(df.columns)}")
    print(f"  Canales (v*): {sum(c.startswith('v') for c in df.columns)}")

    # --- Paso 2: muestras por clase binaria ---
    print("\n--- Paso 2: Muestras (filas) por load_2 ---")
    vc = df_ok["load_2"].value_counts()
    for lab in ["normal", "alta"]:
        n = int(vc.get(lab, 0))
        pct = 100.0 * n / len(df_ok) if len(df_ok) else 0
        print(f"  {lab:8s}: {n:>10,}  ({pct:.1f} % del subconjunto válido)")
    ratio = vc.get("alta", 1) / max(vc.get("normal", 1), 1)
    print(f"  Ratio alta/normal ≈ {ratio:.2f}:1")

    # --- Paso 3: archivos únicos por clase ---
    print("\n--- Paso 3: Archivos .txt únicos por load_2 ---")
    by_file = df_ok.groupby("file_name")["load_2"].first()
    fc = by_file.value_counts()
    for lab in ["normal", "alta"]:
        print(f"  {lab:8s}: {int(fc.get(lab, 0)):>4} archivos")

    # --- Paso 4: por tarea (Arithmetic vs Stroop) ---
    print("\n--- Paso 4: Por tipo de tarea ---")
    for task in sorted(df_ok["task_type"].unique()):
        sub = df_ok[df_ok["task_type"] == task]
        print(f"  {task}:")
        for lab, cnt in sub["load_2"].value_counts().items():
            print(f"        {lab}: {cnt:,}")

    # --- Paso 5: condición original (4 niveles) vs binario ---
    print("\n--- Paso 5: condition_4 vs load_2 (coherencia) ---")
    ct = pd.crosstab(df_ok["condition_4"], df_ok["load_2"], margins=True)
    print(ct.to_string())

    # --- Paso 6: sujetos ---
    print("\n--- Paso 6: Sujetos ---")
    print(f"  subject_id únicos: {df_ok['subject_id'].nunique()}")

    # --- Paso 7: ventanas estimadas (4 s, step 2 s) por archivo ---
    print("\n--- Paso 7: Ventanas de 4 s (step 2 s) estimadas por clase ---")
    wins_normal = 0
    wins_alta = 0
    for (_, g) in df_ok.groupby(
        ["file_name", "subject_id", "task_type", "condition_4", "load_3", "load_2"],
        sort=False,
    ):
        n = len(g)
        w = max(0, (n - WINDOW_SIZE) // STEP_SIZE + 1) if n >= WINDOW_SIZE else 0
        if g["load_2"].iloc[0] == "normal":
            wins_normal += w
        else:
            wins_alta += w
    total_w = wins_normal + wins_alta
    print(f"  normal: ~{wins_normal:,} ventanas")
    print(f"  alta:   ~{wins_alta:,} ventanas")
    print(f"  total:  ~{total_w:,} ventanas")
    if total_w:
        print(
            f"  balance ventanas ≈ {wins_alta / max(wins_normal, 1):.2f}:1 (alta:normal)"
        )

    print("\n" + "=" * 60)
    print(" Siguiente: regenerar features si cambiaste el CSV, luego entrenar con")
    print("   load_2 (scripts eeg_train_window_classifier.py / eeg_train_cnn_lstm.py)")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
