"""
Segmentación de Data-Experimento-Edwin por eventos.

Data-Experimento-Edwin solo tiene carpetas de tarea (*-Laberinto) con AURA_RAW___*.csv.
No hay carpetas "data*" de baseline. Se segmenta cada AURA_RAW por la columna Event
(pre, segment_4, segment_5, ...) igual que en Jeronimo.

Salida: CSVs en formato unificado (timestamp, label, channel_0..channel_7)
en output/edwin_segmented/, listos para análisis posterior.
Los baselines se tomarán del resumen de Rafa (step4) en step_edwin_cognitive_load.
"""

import pandas as pd
import numpy as np
import os
import re
from pathlib import Path

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, 'data', 'Data-Experimento-Edwin')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output', 'edwin_segmented')
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# Canales en AURA_RAW (orden para channel_0..7)
AURA_CHANNELS = ['Fp1', 'Fp2', 'F3', 'Fz', 'F4', 'P3', 'Pz', 'P4']


def parse_aura_time(s):
    """Convierte '[12:41:00.000 20/01/2026]' a segundos desde medianoche."""
    if pd.isna(s) or not isinstance(s, str):
        return np.nan
    m = re.match(r"'?\[(\d{2}):(\d{2}):(\d{2})\.(\d+)\s+(\d{2})/(\d{2})/(\d{4})\]'?", s.strip())
    if not m:
        return np.nan
    h, mi, sec, ms, d, mo, y = m.groups()
    t_sec = int(h) * 3600 + int(mi) * 60 + int(sec) + int(ms[:3]) / 1000.0
    return t_sec


def segment_aura_raw(csv_path):
    """
    Lee AURA_RAW, segmenta por columna Event y devuelve DataFrame con
    timestamp, label (pre, segment_4, segment_5, ...), channel_0..7.
    """
    df = pd.read_csv(csv_path, skiprows=[1])
    evcol = None
    for c in df.columns:
        c_clean = c.strip("'\"")
        if 'event' in c_clean.lower() or 'button' in c_clean.lower():
            evcol = c
            break
    if evcol is None:
        evcol = df.columns[-1]

    timecol = df.columns[0]
    times = df[timecol].apply(parse_aura_time)
    t0 = times.min()
    if np.isnan(t0):
        t0 = 0
    timestamp = times - t0

    channel_cols = []
    for ch in AURA_CHANNELS:
        for c in df.columns:
            if c.strip("'\"") == ch:
                channel_cols.append(c)
                break
        else:
            channel_cols.append(None)
    if len(channel_cols) != 8 or any(c is None for c in channel_cols):
        channel_cols = [df.columns[i] for i in range(1, 9)]

    events = df[evcol].values
    event_values = sorted([x for x in np.unique(events) if x != 0])
    boundaries = {}
    for ev in event_values:
        idx = np.where(events == ev)[0]
        if len(idx) > 0:
            boundaries[ev] = int(idx[0])
    sorted_evs = sorted(boundaries.keys(), key=lambda e: boundaries[e])
    n = len(df)
    labels = np.array(['pre'] * n)
    for i, ev in enumerate(sorted_evs):
        start = boundaries[ev]
        end = boundaries[sorted_evs[i + 1]] if i + 1 < len(sorted_evs) else n
        labels[start:end] = f'segment_{ev}'

    out = pd.DataFrame({'timestamp': timestamp})
    out['label'] = labels
    for i, col in enumerate(channel_cols):
        out[f'channel_{i}'] = df[col].values
    return out, sorted_evs, boundaries


def process_task_folder(folder_path, session_name, output_dir):
    """Lee AURA_RAW___*.csv, segmenta por eventos y guarda CSV unificado."""
    csvs = list(Path(folder_path).glob('AURA_RAW___*.csv'))
    if not csvs:
        return None
    csv_path = csvs[0]
    try:
        df_out, sorted_evs, boundaries = segment_aura_raw(csv_path)
    except Exception as e:
        print(f"  ERROR en {session_name}: {e}")
        return None
    safe_name = re.sub(r'[^\w\-]', '_', session_name)
    out_path = os.path.join(output_dir, f'edwin_task_{safe_name}.csv')
    df_out.to_csv(out_path, index=False)
    print(f"  Task: {out_path} ({len(df_out)} filas, segmentos: {sorted_evs})")
    return out_path


def main():
    print("="*80)
    print("SEGMENTACION DATA-EXPERIMENTO-EDWIN POR EVENTOS")
    print("="*80)
    print(f"Directorio datos: {DATA_DIR}")
    print(f"Salida: {OUTPUT_DIR}")
    print("(Solo tareas Laberinto; baselines se usan desde Rafa en step_edwin_cognitive_load)")
    print()

    if not os.path.isdir(DATA_DIR):
        print(f"ERROR: No existe {DATA_DIR}")
        return

    data_path = Path(DATA_DIR)
    task_count = 0

    for folder in sorted(data_path.iterdir()):
        if not folder.is_dir():
            continue
        name = folder.name
        process_task_folder(folder, name, OUTPUT_DIR)
        task_count += 1

    print()
    print("="*80)
    print("COMPLETADO")
    print("="*80)
    print(f"Tareas procesadas: {task_count}")
    print(f"CSVs guardados en: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
