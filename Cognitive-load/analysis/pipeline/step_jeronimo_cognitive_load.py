"""
Análisis de cognitive load (ratio Theta/Alpha) en Data-Experimento-Jeronimo.

- Lee los CSVs segmentados en output/jeronimo_segmented/
- Calcula ratio Theta(Fz)/Alpha(Pz) por fase (baseline, pre, segment_4, segment_5, ...)
- Compara cada etapa de tarea con el baseline del mismo sujeto
- Salida: CSV resumen + gráficas comparando todas las etapas vs baseline
"""

import pandas as pd
import numpy as np
from scipy import signal
from pathlib import Path
import os
import re
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SEGMENTED_DIR = os.path.join(BASE_DIR, 'output', 'jeronimo_segmented')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output', 'jeronimo_analysis')
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# Sujetos en Jeronimo sin baseline en jeronimo_segmented -> usar baseline de Data-Experimento-Rafa
TASK_KEY_TO_RAFA_BASELINE = {'eli': 'eliza', 'jeronimo': 'Jeronimo'}
# Resumen de step4 (Rafa) para leer baseline eyes closed/open ya calculado
RAFA_SUMMARY_CSV = os.path.join(BASE_DIR, 'output', 'analysis_output', 'cognitive_load_cleaned_summary.csv')

SAMPLE_RATE = 250
THETA_BAND = (4.0, 7.0)
ALPHA_BAND = (8.0, 12.0)
FZ_CHANNEL = 3
PZ_CHANNEL = 6
WINDOW_SAMPLES = 250


def apply_bandpass_filter(signal_data, low_freq=1.0, high_freq=40.0, sample_rate=250):
    if len(signal_data) < 3:
        return signal_data
    nyquist = sample_rate / 2
    low = low_freq / nyquist
    high = high_freq / nyquist
    if low >= 1.0 or high >= 1.0 or low <= 0 or high <= 0:
        return signal_data
    b, a = signal.butter(4, [low, high], btype='band')
    return signal.filtfilt(b, a, signal_data)


def apply_notch_filter(signal_data, notch_freq=60.0, sample_rate=250, Q=30.0):
    if len(signal_data) < 3:
        return signal_data
    nyquist = sample_rate / 2
    w0 = notch_freq / nyquist
    if w0 >= 1.0 or w0 <= 0:
        return signal_data
    b, a = signal.iirnotch(w0, Q)
    return signal.filtfilt(b, a, signal_data)


def apply_car(eeg_matrix):
    n_samples, n_channels = eeg_matrix.shape
    car_ref = np.mean(eeg_matrix, axis=1, keepdims=True)
    return eeg_matrix - car_ref


def calculate_bandpower(signal_data, freq_band, sample_rate):
    if len(signal_data) < sample_rate // 2:
        return 0.0
    nperseg = min(len(signal_data), sample_rate)
    freqs, psd = signal.welch(signal_data, sample_rate, nperseg=nperseg, noverlap=nperseg//2)
    idx_band = np.logical_and(freqs >= freq_band[0], freqs <= freq_band[1])
    if np.sum(idx_band) == 0:
        return 0.0
    return np.trapezoid(psd[idx_band], freqs[idx_band])


def preprocess_signal(channel_data, sample_rate=250):
    notch = apply_notch_filter(channel_data, sample_rate=sample_rate)
    return apply_bandpass_filter(notch, sample_rate=sample_rate)


def compute_ratio_for_phase(phase_data, sample_rate=250):
    """
    phase_data: DataFrame con columnas channel_0..channel_7, ordenado por timestamp.
    Devuelve (mean_ratio, n_windows) o (np.nan, 0) si no hay suficientes datos.
    """
    if phase_data is None or len(phase_data) < sample_rate // 2:
        return np.nan, 0
    phase_data = phase_data.sort_values('timestamp')
    n_samples = len(phase_data)
    eeg = np.zeros((n_samples, 8))
    for ch in range(8):
        eeg[:, ch] = phase_data[f'channel_{ch}'].values
    filtered = np.zeros_like(eeg)
    for ch in range(8):
        filtered[:, ch] = preprocess_signal(eeg[:, ch], sample_rate)
    eeg_car = apply_car(filtered)
    fz = eeg_car[:, FZ_CHANNEL]
    pz = eeg_car[:, PZ_CHANNEL]
    ratios = []
    step = WINDOW_SAMPLES // 2
    for start in range(0, len(fz) - WINDOW_SAMPLES + 1, step):
        fz_win = fz[start:start + WINDOW_SAMPLES]
        pz_win = pz[start:start + WINDOW_SAMPLES]
        theta = calculate_bandpower(fz_win, THETA_BAND, sample_rate)
        alpha = calculate_bandpower(pz_win, ALPHA_BAND, sample_rate)
        if alpha > 0 and np.isfinite(theta) and np.isfinite(alpha):
            r = theta / alpha
            if np.isfinite(r) and 0.01 < r < 100:
                ratios.append(r)
    if not ratios:
        return np.nan, 0
    return np.mean(ratios), len(ratios)


def baseline_subject_key(baseline_filename):
    """De jeronimo_baseline_data_dani-jeronimo.csv -> 'dani'; data_Joss-Jeronimo -> 'joss'; data1-jero -> '1'."""
    name = Path(baseline_filename).stem
    name = name.replace('jeronimo_baseline_', '')
    if name.startswith('data_'):
        name = name[5:]  # dani-jeronimo, Joss-Jeronimo, ...
    elif name.startswith('data'):
        name = name.replace('data', '').replace('-jero', '').strip('_')
    name = re.sub(r'-jero(nimo)?$', '', name, flags=re.IGNORECASE)
    return name.strip('- _').lower()


def task_subject_key(task_filename):
    """De jeronimo_task_Dani-CanineQuest.csv -> 'dani'."""
    name = Path(task_filename).stem.replace('jeronimo_task_', '')
    part = name.split('-')[0].split('_')[0]
    return part.lower()


def load_baseline_from_rafa_summary(task_key, rafa_subject_name):
    """
    Lee baseline (mean_ratio_after) desde cognitive_load_cleaned_summary.csv de step4
    para sujeto rafa_subject_name (eliza, Jeronimo). Preferencia: baseline_eyes_closed.
    """
    if not os.path.isfile(RAFA_SUMMARY_CSV):
        return None
    df = pd.read_csv(RAFA_SUMMARY_CSV)
    if 'subject' not in df.columns or 'phase' not in df.columns or 'mean_ratio_after' not in df.columns:
        return None
    subj = df[df['subject'] == rafa_subject_name]
    if len(subj) == 0:
        return None
    for phase in ('baseline_eyes_closed', 'baseline_eyes_open'):
        row = subj[subj['phase'] == phase]
        if len(row) > 0:
            r = row.iloc[0]['mean_ratio_after']
            if np.isfinite(r) and r > 0:
                return float(r)
    return None


def load_baseline_ratios(segmented_dir):
    """
    Carga baselines: (1) jeronimo_segmented baseline_*.csv,
    (2) para Eli y Jeronimo sin baseline en Jeronimo, usa Data-Experimento-Rafa (eyes closed/open).
    Devuelve dict subject_key -> mean_ratio.
    """
    baseline_dir = Path(segmented_dir)
    files = list(baseline_dir.glob('jeronimo_baseline_*.csv'))
    result = {}
    for f in files:
        key = baseline_subject_key(f.name)
        df = pd.read_csv(f)
        if 'label' not in df.columns and 'phase_label' in df.columns:
            df['label'] = df['phase_label']
        phase_data = df
        mean_r, n_win = compute_ratio_for_phase(phase_data)
        result[key] = mean_r
        print(f"  Baseline {key} (Jeronimo): mean_ratio={mean_r:.4f}, n_windows={n_win}")
    # Sujetos en Jeronimo sin baseline en jeronimo -> usar baseline de Data-Experimento-Rafa (step4 summary)
    for task_key, rafa_name in TASK_KEY_TO_RAFA_BASELINE.items():
        if task_key not in result:
            ratio = load_baseline_from_rafa_summary(task_key, rafa_name)
            if ratio is not None and np.isfinite(ratio) and ratio > 0:
                result[task_key] = ratio
                print(f"  Baseline {task_key} (desde Rafa step4, {rafa_name}): mean_ratio={ratio:.4f}")
    return result


def analyze_task_file(csv_path, baseline_ratios):
    """Analiza un CSV de tarea: ratio por cada label (pre, segment_4, ...)."""
    df = pd.read_csv(csv_path)
    session_name = Path(csv_path).stem.replace('jeronimo_task_', '')
    subject_key = task_subject_key(csv_path)
    baseline_ratio = baseline_ratios.get(subject_key)

    rows = []
    for label in df['label'].unique():
        phase_data = df[df['label'] == label]
        mean_r, n_win = compute_ratio_for_phase(phase_data)
        row = {
            'session': session_name,
            'phase': label,
            'mean_ratio': mean_r,
            'n_windows': n_win,
            'baseline_ratio': baseline_ratio if baseline_ratio is not None and np.isfinite(baseline_ratio) else np.nan,
        }
        if baseline_ratio is not None and np.isfinite(baseline_ratio) and baseline_ratio > 0 and np.isfinite(mean_r):
            row['normalized_ratio'] = mean_r / baseline_ratio
            row['relative_change_pct'] = 100 * (mean_r - baseline_ratio) / baseline_ratio
        else:
            row['normalized_ratio'] = np.nan
            row['relative_change_pct'] = np.nan
        rows.append(row)
    return rows


def create_comparison_plot(df_summary, output_dir):
    """Gráfica: por sesión, barras de mean_ratio por fase y línea de baseline."""
    sessions = df_summary['session'].unique()
    n_sessions = len(sessions)
    if n_sessions == 0:
        return
    n_cols = 4
    n_rows = (n_sessions + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    for idx, session in enumerate(sessions):
        row, col = idx // n_cols, idx % n_cols
        ax = axes[row, col]
        data = df_summary[df_summary['session'] == session]
        phase_order = sorted(data['phase'].unique(), key=lambda x: (0 if x == 'pre' else 1, x))
        data = data.set_index('phase').loc[phase_order].reset_index()
        phases = data['phase'].tolist()
        means = data['mean_ratio'].tolist()
        baseline_val = data['baseline_ratio'].iloc[0] if 'baseline_ratio' in data.columns else np.nan
        x = np.arange(len(phases))
        colors = ['#81c784' if p == 'pre' else '#ffb74d' if p.startswith('segment') else '#4fc3f7' for p in phases]
        ax.bar(x, means, color=colors, alpha=0.9, edgecolor='white')
        if np.isfinite(baseline_val):
            ax.axhline(y=baseline_val, color='red', linestyle='--', linewidth=1.5, label='Baseline')
        ax.set_xticks(x)
        ax.set_xticklabels(phases, rotation=45, ha='right', fontsize=8)
        ax.set_ylabel('Theta/Alpha ratio')
        ax.set_title(session[:25] + ('...' if len(session) > 25 else ''), fontsize=9)
        if np.isfinite(baseline_val):
            ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_facecolor('#fafafa')
    for idx in range(len(sessions), n_rows * n_cols):
        row, col = idx // n_cols, idx % n_cols
        axes[row, col].set_visible(False)
    plt.suptitle('Cognitive load por fase vs baseline (Jeronimo)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    out_path = os.path.join(output_dir, 'jeronimo_cognitive_load_by_session.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Grafico guardado: {out_path}")


def create_normalized_plot(df_summary, output_dir):
    """Gráfica: ratio normalizado (baseline=1) por fase y sesión."""
    df_n = df_summary.dropna(subset=['normalized_ratio']).copy()
    if len(df_n) == 0:
        return
    sessions = df_n['session'].unique()
    phases_all = sorted(df_n['phase'].unique(), key=lambda x: (0 if x == 'pre' else 1, x))
    x = np.arange(len(phases_all))
    width = 0.8 / max(len(sessions), 1)
    fig, ax = plt.subplots(figsize=(14, 6))
    for i, session in enumerate(sessions):
        data = df_n[df_n['session'] == session]
        vals = []
        for p in phases_all:
            row = data[data['phase'] == p]
            vals.append(row['normalized_ratio'].iloc[0] if len(row) else np.nan)
        offset = (i - len(sessions) / 2 + 0.5) * width
        ax.bar(x + offset, vals, width, label=session[:20], alpha=0.8)
    ax.axhline(y=1.0, color='red', linestyle='--', linewidth=2, label='Baseline = 1')
    ax.set_xticks(x)
    ax.set_xticklabels(phases_all, rotation=45, ha='right')
    ax.set_ylabel('Ratio normalizado (Baseline = 1)')
    ax.set_title('Cognitive load: todas las etapas vs baseline (Jeronimo)')
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=7)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    out_path = os.path.join(output_dir, 'jeronimo_cognitive_load_normalized.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Grafico guardado: {out_path}")


def main():
    print("="*80)
    print("ANALISIS COGNITIVE LOAD - DATA-EXPERIMENTO-JERONIMO")
    print("="*80)
    print(f"Entrada: {SEGMENTED_DIR}")
    print(f"Salida: {OUTPUT_DIR}")
    print()

    if not os.path.isdir(SEGMENTED_DIR):
        print(f"ERROR: No existe {SEGMENTED_DIR}. Ejecuta antes step_jeronimo_segment_by_events.py")
        return

    print("Cargando baselines...")
    baseline_ratios = load_baseline_ratios(SEGMENTED_DIR)
    print()

    task_files = list(Path(SEGMENTED_DIR).glob('jeronimo_task_*.csv'))
    all_rows = []
    for tf in sorted(task_files):
        session_name = tf.stem.replace('jeronimo_task_', '')
        print(f"Analizando tarea: {session_name}")
        rows = analyze_task_file(str(tf), baseline_ratios)
        for r in rows:
            print(f"  {r['phase']}: mean_ratio={r['mean_ratio']:.4f}, n_windows={r['n_windows']}, normalized={r.get('normalized_ratio', np.nan):.3f}")
        all_rows.extend(rows)

    if not all_rows:
        print("No hay datos para resumir.")
        return

    df_summary = pd.DataFrame(all_rows)
    csv_path = os.path.join(OUTPUT_DIR, 'jeronimo_cognitive_load_summary.csv')
    df_summary.to_csv(csv_path, index=False)
    print(f"\nResumen guardado: {csv_path}")

    create_comparison_plot(df_summary, OUTPUT_DIR)
    create_normalized_plot(df_summary, OUTPUT_DIR)

    print("\n" + "="*80)
    print("COMPLETADO")
    print("="*80)


if __name__ == '__main__':
    main()
