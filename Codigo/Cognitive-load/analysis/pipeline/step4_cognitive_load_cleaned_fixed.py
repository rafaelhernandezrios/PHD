"""
Paso 4 (CORREGIDO): Cálculo de Cognitive Load con datos limpiados
- Usa la frecuencia de muestreo REAL estimada desde los timestamps,
  tanto para filtrado como para Welch/bandpower.
- Ventana en SEGUNDOS (no en muestras a 250 Hz), para que el tamaño
  espectral sea consistente entre grabaciones a distintas SR.
- Criterio de fase corta también en SEGUNDOS.
- Escribe a un subdirectorio `analysis_output_fixed` para no sobrescribir
  los resultados originales (buggy).

Cambio crítico respecto al script original:
    En la versión anterior, `calculate_cognitive_load_by_phase(..., SAMPLE_RATE)`
    recibía SAMPLE_RATE=250 hardcoded. Welch usaba ese fs para mapear bins,
    por lo que "theta 4-7 Hz" y "alpha 8-12 Hz" se leían en ubicaciones
    equivocadas del espectro cuando la SR real era ~50 Hz (AURA).
    Esta corrección pasa `actual_sample_rate` en su lugar.
"""

import pandas as pd
import numpy as np
from scipy import signal, stats
from scipy.interpolate import interp1d
from pathlib import Path
import os
import matplotlib.pyplot as plt

# Configuración
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, 'data', 'Data-Experimento-Rafa')
ELECTRON_DATA_DIR = os.path.join(BASE_DIR, 'electron')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output', 'analysis_output_fixed')
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

USE_FZ_PZ = True
OUTPUT_SUFFIX = "_fz_pz" if USE_FZ_PZ else "_region"

# Parámetros (ahora en SEGUNDOS, no en muestras)
DEFAULT_SAMPLE_RATE = 250   # fallback si no hay timestamps
THETA_BAND = (4.0, 7.0)     # Hz
ALPHA_BAND = (8.0, 12.0)    # Hz
WINDOW_SECONDS = 1.0        # ventana de 1 s
STEP_SECONDS   = 0.5        # solapamiento 50%

FZ_CHANNEL = 3
PZ_CHANNEL = 6
F_CHANNELS = [0, 1, 2, 3, 4]
P_CHANNELS = [5, 6, 7]
BAD_CHANNELS = []

CHANNEL_NAMES = {0: 'Fp1', 1: 'Fp2', 2: 'F3', 3: 'Fz',
                 4: 'F4', 5: 'P3', 6: 'Pz', 7: 'P4'}

LABELS_OF_INTEREST = [
    'baseline_eyes_open',
    'baseline_eyes_closed',
    'low_cognitive_load',
    'high_cognitive_load',
]

Z_SCORE_THRESHOLD = 3.0
IQR_MULTIPLIER = 3.0
AMPLITUDE_THRESHOLD = 200  # uV

# Criterios de exclusión
ARTIFACT_PCT_THRESHOLD = 50.0
MIN_WINDOWS_AFTER      = 2
MIN_PHASE_SECONDS      = 32.0   # ~32 s, equivalente al viejo 8000@250Hz


def get_included_subjects(df_summary):
    """Sujetos incluibles: fases low y high suficientemente largas, pocos artefactos,
    y suficientes ventanas después de limpieza."""
    included, excluded = [], {}
    for subject in df_summary['subject'].unique():
        subject_data = df_summary[df_summary['subject'] == subject]
        low_data = subject_data[subject_data['phase'] == 'low_cognitive_load']
        high_data = subject_data[subject_data['phase'] == 'high_cognitive_load']
        if len(low_data) == 0 or len(high_data) == 0:
            excluded[subject] = "falta fase low o high"
            continue
        low_row, high_row = low_data.iloc[0], high_data.iloc[0]
        reasons = []
        if low_row['phase_seconds'] < MIN_PHASE_SECONDS or high_row['phase_seconds'] < MIN_PHASE_SECONDS:
            reasons.append(f"fase corta (low={low_row['phase_seconds']:.1f}s, high={high_row['phase_seconds']:.1f}s)")
        if low_row['fz_artifact_pct'] > ARTIFACT_PCT_THRESHOLD or low_row['pz_artifact_pct'] > ARTIFACT_PCT_THRESHOLD:
            reasons.append(f"low artifacts Fz={low_row['fz_artifact_pct']:.1f}% Pz={low_row['pz_artifact_pct']:.1f}%")
        if high_row['fz_artifact_pct'] > ARTIFACT_PCT_THRESHOLD or high_row['pz_artifact_pct'] > ARTIFACT_PCT_THRESHOLD:
            reasons.append(f"high artifacts Fz={high_row['fz_artifact_pct']:.1f}% Pz={high_row['pz_artifact_pct']:.1f}%")
        if low_row['n_windows_after'] < MIN_WINDOWS_AFTER:
            reasons.append(f"low ventanas={int(low_row['n_windows_after'])}")
        if high_row['n_windows_after'] < MIN_WINDOWS_AFTER:
            reasons.append(f"high ventanas={int(high_row['n_windows_after'])}")
        if reasons:
            excluded[subject] = "; ".join(reasons)
        else:
            included.append(subject)
    return included, excluded


def apply_bandpass_filter(x, low_freq=1.0, high_freq=40.0, sample_rate=250):
    if len(x) < 3:
        return x
    if sample_rate < high_freq * 2:
        return x
    nyq = sample_rate / 2
    lo, hi = low_freq / nyq, high_freq / nyq
    if lo <= 0 or hi >= 1.0 or lo >= hi:
        return x
    b, a = signal.butter(4, [lo, hi], btype='band')
    return signal.filtfilt(b, a, x)


def apply_notch_filter(x, notch_freq=60.0, sample_rate=250, Q=30.0):
    if len(x) < 3:
        return x
    if sample_rate < notch_freq * 2:
        return x
    nyq = sample_rate / 2
    w0 = notch_freq / nyq
    if w0 <= 0 or w0 >= 1.0:
        return x
    b, a = signal.iirnotch(w0, Q)
    return signal.filtfilt(b, a, x)


def apply_car(eeg, bad_channel_idx=None, bad_channel_indices=None):
    n_samples, n_channels = eeg.shape
    exclude = list(bad_channel_indices) if bad_channel_indices else []
    if bad_channel_idx is not None:
        exclude.append(bad_channel_idx)
    valid = [i for i in range(n_channels) if i not in exclude] if exclude else list(range(n_channels))
    if not valid:
        return eeg
    ref = np.mean(eeg[:, valid], axis=1, keepdims=True)
    return eeg - ref


def get_region_signal(eeg_car, channel_indices, exclude_indices):
    use = [i for i in channel_indices if i not in exclude_indices]
    if not use:
        return None
    return np.mean(eeg_car[:, use], axis=1)


def detect_artifacts_combined(x, z_threshold=Z_SCORE_THRESHOLD,
                              iqr_multiplier=IQR_MULTIPLIER,
                              amp_threshold=AMPLITUDE_THRESHOLD):
    if len(x) < 3:
        return np.zeros(len(x), dtype=bool), 0
    z = np.abs(stats.zscore(x))
    mz = z > z_threshold
    if len(x) >= 4:
        q1, q3 = np.percentile(x, [25, 75])
        iqr = q3 - q1
        lb, ub = q1 - iqr_multiplier * iqr, q3 + iqr_multiplier * iqr
        mi = (x < lb) | (x > ub)
    else:
        mi = np.zeros(len(x), dtype=bool)
    ma = np.abs(x) > amp_threshold
    m = mz | mi | ma
    return m, int(np.sum(m))


def remove_artifacts_interpolation(x, mask):
    if np.sum(mask) == 0:
        return x.copy()
    y = x.copy()
    valid_idx = np.where(~mask)[0]
    bad_idx = np.where(mask)[0]
    if len(valid_idx) < 2:
        valid_data = x[~mask]
        if len(valid_data) > 0:
            lb, ub = np.percentile(valid_data, [1, 99])
            y[bad_idx] = np.clip(y[bad_idx], lb, ub)
        return y
    f = interp1d(valid_idx, y[valid_idx], kind='linear',
                 fill_value='extrapolate', bounds_error=False)
    y[bad_idx] = f(bad_idx)
    return y


def calculate_bandpower(x, band, sample_rate):
    """Bandpower usando Welch, con nperseg adaptado a la SR real."""
    if sample_rate <= 0:
        return 0.0
    if len(x) < max(8, int(sample_rate / 2)):
        return 0.0
    # nperseg ≈ 1 segundo, acotado al tamaño disponible
    nperseg = max(8, min(len(x), int(sample_rate)))
    freqs, psd = signal.welch(x, fs=sample_rate, nperseg=nperseg,
                              noverlap=nperseg // 2)
    idx = np.logical_and(freqs >= band[0], freqs <= band[1])
    if np.sum(idx) == 0:
        return 0.0
    return float(np.trapezoid(psd[idx], freqs[idx]))


def preprocess_signal(x, sample_rate):
    """Notch 60 + bandpass 1-40 a la SR real (si la SR lo permite)."""
    fs = sample_rate if sample_rate >= 10 else DEFAULT_SAMPLE_RATE
    y = apply_notch_filter(x, sample_rate=fs)
    y = apply_bandpass_filter(y, sample_rate=fs)
    return y


def calculate_cognitive_load_by_phase(fz, pz, sample_rate,
                                      window_seconds=WINDOW_SECONDS,
                                      step_seconds=STEP_SECONDS):
    ratios, thetas, alphas = [], [], []
    if sample_rate <= 0:
        return ratios, thetas, alphas
    win = max(4, int(round(window_seconds * sample_rate)))
    step = max(1, int(round(step_seconds * sample_rate)))
    if len(fz) < win or len(pz) < win:
        # si la fase es demasiado corta usa la señal completa como una sola ventana
        if len(fz) >= max(8, int(sample_rate / 2)):
            win = min(len(fz), len(pz))
            step = win
        else:
            return ratios, thetas, alphas
    for s in range(0, len(fz) - win + 1, step):
        e = s + win
        tp = calculate_bandpower(fz[s:e], THETA_BAND, sample_rate)
        ap = calculate_bandpower(pz[s:e], ALPHA_BAND, sample_rate)
        if ap > 0 and np.isfinite(tp) and np.isfinite(ap):
            r = tp / ap
            if np.isfinite(r) and 0.01 < r < 100:
                ratios.append(r)
                thetas.append(tp)
                alphas.append(ap)
    return ratios, thetas, alphas


def analyze_subject_cleaned(csv_path):
    subject_name = Path(csv_path).parent.name.replace('data_', '')
    print(f"\n{'='*80}\nANALISIS (FIX): {subject_name.upper()}\n{'='*80}")
    df = pd.read_csv(csv_path)
    print(f"Total de muestras: {len(df):,}")

    # SR real a partir de timestamps (ya usada para filtrado en el script original)
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')
        df_sorted = df.sort_values('timestamp')
        median_interval = df_sorted['timestamp'].diff().dropna().median()
        actual_sr = 1.0 / median_interval if median_interval and median_interval > 0 else DEFAULT_SAMPLE_RATE
    else:
        actual_sr = DEFAULT_SAMPLE_RATE
    print(f"SR real estimada: {actual_sr:.2f} Hz")

    results = {'subject': subject_name, 'actual_sample_rate': float(actual_sr), 'phases': {}}

    for label in LABELS_OF_INTEREST:
        phase_df = df[df['label'] == label].copy()
        if len(phase_df) == 0:
            continue
        if 'timestamp' in phase_df.columns:
            phase_df = phase_df.sort_values('timestamp')
        n_samples = len(phase_df)
        phase_seconds = n_samples / actual_sr if actual_sr > 0 else 0.0
        print(f"\n  Fase: {label}  |  n={n_samples:,}  ({phase_seconds:.1f} s)")

        eeg = np.zeros((n_samples, 8))
        for ch in range(8):
            eeg[:, ch] = phase_df[f'channel_{ch}'].values

        # Preprocesado a la SR real
        eeg_f = np.zeros_like(eeg)
        for ch in range(8):
            eeg_f[:, ch] = preprocess_signal(eeg[:, ch], actual_sr)

        eeg_car = apply_car(eeg_f, bad_channel_indices=BAD_CHANNELS if not USE_FZ_PZ else None)
        if USE_FZ_PZ:
            fz = eeg_car[:, FZ_CHANNEL].copy()
            pz = eeg_car[:, PZ_CHANNEL].copy()
        else:
            fz = get_region_signal(eeg_car, F_CHANNELS, BAD_CHANNELS)
            pz = get_region_signal(eeg_car, P_CHANNELS, BAD_CHANNELS)
            if fz is None or pz is None:
                print("    OMITIDO: región F o P sin canales válidos")
                continue

        fz_mask, fz_n = detect_artifacts_combined(fz)
        pz_mask, pz_n = detect_artifacts_combined(pz)
        fz_pct = 100 * fz_n / len(fz) if len(fz) else 0.0
        pz_pct = 100 * pz_n / len(pz) if len(pz) else 0.0

        fz_clean = remove_artifacts_interpolation(fz, fz_mask)
        pz_clean = remove_artifacts_interpolation(pz, pz_mask)

        # *** AQUI EL FIX: pasar actual_sr en vez de SAMPLE_RATE=250 ***
        ratios_before, t_b, a_b = calculate_cognitive_load_by_phase(fz, pz, actual_sr)
        ratios_after,  t_a, a_a = calculate_cognitive_load_by_phase(fz_clean, pz_clean, actual_sr)

        results['phases'][label] = {
            'n_samples': int(n_samples),
            'phase_seconds': float(phase_seconds),
            'fz_artifacts': int(fz_n), 'pz_artifacts': int(pz_n),
            'fz_artifact_pct': float(fz_pct), 'pz_artifact_pct': float(pz_pct),
            'n_windows_before': len(ratios_before), 'n_windows_after': len(ratios_after),
            'mean_ratio_before': float(np.mean(ratios_before)) if ratios_before else 0.0,
            'mean_ratio_after':  float(np.mean(ratios_after))  if ratios_after  else 0.0,
            'median_ratio_before': float(np.median(ratios_before)) if ratios_before else 0.0,
            'median_ratio_after':  float(np.median(ratios_after))  if ratios_after  else 0.0,
            'std_ratio_before':  float(np.std(ratios_before)) if ratios_before else 0.0,
            'std_ratio_after':   float(np.std(ratios_after))  if ratios_after  else 0.0,
            'mean_theta_after':  float(np.mean(t_a)) if t_a else 0.0,
            'mean_alpha_after':  float(np.mean(a_a)) if a_a else 0.0,
            'ratios_before': ratios_before,
            'ratios_after':  ratios_after,
        }
        print(f"    Artefactos Fz: {fz_n} ({fz_pct:.1f}%)  Pz: {pz_n} ({pz_pct:.1f}%)")
        print(f"    Ventanas: before={len(ratios_before)}  after={len(ratios_after)}")
        print(f"    Ratio medio after: {results['phases'][label]['mean_ratio_after']:.3f}")
    return results


def normalize_by_baseline(all_results):
    out = []
    for r in all_results:
        rr = r.copy(); rr['phases_normalized'] = {}
        base = None
        if 'baseline_eyes_closed' in r['phases']:
            base = r['phases']['baseline_eyes_closed']['mean_ratio_after']
        elif 'baseline_eyes_open' in r['phases']:
            base = r['phases']['baseline_eyes_open']['mean_ratio_after']
        if base is None or not np.isfinite(base) or base <= 0:
            base_o = r['phases'].get('baseline_eyes_open', {}).get('mean_ratio_after')
            base_c = r['phases'].get('baseline_eyes_closed', {}).get('mean_ratio_after')
            bs = [b for b in (base_o, base_c) if b is not None and np.isfinite(b) and b > 0]
            base = float(np.mean(bs)) if bs else None
        for phase, pd_ in r['phases'].items():
            np_ = pd_.copy()
            v = pd_['mean_ratio_after']
            if base and base > 0 and np.isfinite(v):
                np_['normalized_ratio'] = float(v / base)
                np_['baseline_reference'] = float(base)
            else:
                np_['normalized_ratio'] = np.nan
                np_['baseline_reference'] = float(base) if base is not None else np.nan
            rr['phases_normalized'][phase] = np_
        out.append(rr)
    return out


def create_final_report(all_results, output_dir):
    print(f"\n{'='*80}\nREPORTE FINAL (FIX)\n{'='*80}")
    rows = []
    for r in all_results:
        for phase in LABELS_OF_INTEREST:
            if phase in r['phases']:
                p = r['phases'][phase]
                rows.append({
                    'subject': r['subject'], 'phase': phase,
                    'actual_sample_rate': r['actual_sample_rate'],
                    'n_samples': p['n_samples'], 'phase_seconds': p['phase_seconds'],
                    'fz_artifacts': p['fz_artifacts'], 'pz_artifacts': p['pz_artifacts'],
                    'fz_artifact_pct': p['fz_artifact_pct'], 'pz_artifact_pct': p['pz_artifact_pct'],
                    'n_windows_before': p['n_windows_before'],
                    'n_windows_after':  p['n_windows_after'],
                    'mean_ratio_before': p['mean_ratio_before'],
                    'mean_ratio_after':  p['mean_ratio_after'],
                    'median_ratio_before': p['median_ratio_before'],
                    'median_ratio_after':  p['median_ratio_after'],
                    'std_ratio_before':  p['std_ratio_before'],
                    'std_ratio_after':   p['std_ratio_after'],
                    'mean_theta_after':  p['mean_theta_after'],
                    'mean_alpha_after':  p['mean_alpha_after'],
                })
    if not rows:
        print("Sin datos"); return pd.DataFrame()
    df = pd.DataFrame(rows)
    out = os.path.join(output_dir, 'cognitive_load_cleaned_summary' + OUTPUT_SUFFIX + '.csv')
    df.to_csv(out, index=False)
    print(f"Guardado: {out}")

    # Tabla en pantalla
    print("\nRESUMEN (mean_ratio_after):")
    print("-" * 100)
    print(f"{'Sujeto':<12}{'SR(Hz)':>8}{'Open':>10}{'Closed':>10}{'Low':>10}{'High':>10}  {'Berger?':>9}  {'H>L?':>6}")
    print("-" * 100)
    for subj in df['subject'].unique():
        d = df[df['subject'] == subj]
        sr = d['actual_sample_rate'].iloc[0]
        def g(p):
            row = d[d['phase'] == p]
            return row['mean_ratio_after'].iloc[0] if len(row) else np.nan
        o, c, lo, hi = g('baseline_eyes_open'), g('baseline_eyes_closed'), g('low_cognitive_load'), g('high_cognitive_load')
        # Para el efecto Berger el marcador correcto es alpha_Pz cerrado > abierto; aquí
        # miramos alpha_after directamente si está disponible.
        def ga(p):
            row = d[d['phase'] == p]
            return row['mean_alpha_after'].iloc[0] if len(row) else np.nan
        a_open, a_closed = ga('baseline_eyes_open'), ga('baseline_eyes_closed')
        berger = (np.isfinite(a_open) and np.isfinite(a_closed) and a_closed > a_open)
        hlflag = (np.isfinite(lo) and np.isfinite(hi) and hi > lo)
        def fmt(x): return f"{x:.3f}" if np.isfinite(x) else "  N/A"
        print(f"{subj:<12}{sr:>8.1f}{fmt(o):>10}{fmt(c):>10}{fmt(lo):>10}{fmt(hi):>10}  {'[OK]' if berger else '[X] ':>9}  {'[OK]' if hlflag else '[X] ':>6}")

    included, excluded = get_included_subjects(df)
    print(f"\nSujetos incluibles: {len(included)}  |  Excluidos: {len(excluded)}")
    if excluded:
        for s, reason in sorted(excluded.items()):
            print(f"  - {s}: {reason}")
    return df


def create_charts(all_results, output_dir):
    """Gráficas resumen comparables a las originales, con la SR real."""
    subjects = [r['subject'] for r in all_results]
    phases = LABELS_OF_INTEREST
    phase_labels = ['Baseline Open', 'Baseline Closed', 'Low Load', 'High Load']

    data_abs = {pl: [] for pl in phase_labels}
    for r in all_results:
        for ph, pl in zip(phases, phase_labels):
            v = r['phases'].get(ph, {}).get('mean_ratio_after', np.nan)
            data_abs[pl].append(v if np.isfinite(v) else np.nan)

    x = np.arange(len(subjects))
    width = 0.2
    colors = ['#4fc3f7', '#81c784', '#ffb74d', '#e57373']

    fig, ax = plt.subplots(figsize=(16, 8))
    fig.suptitle('Cognitive Load Ratio (FIX: SR real) — Todos los Sujetos', fontsize=14, fontweight='bold', color='white')
    for i, (pl, color) in enumerate(zip(phase_labels, colors)):
        off = (i - 1.5) * width
        vals = [v if not np.isnan(v) else 0 for v in data_abs[pl]]
        ax.bar(x + off, vals, width, label=pl, color=color, alpha=0.9, edgecolor='white', linewidth=0.5)
    ax.set_xlabel('Sujeto', color='white'); ax.set_ylabel('Theta(Fz)/Alpha(Pz)', color='white')
    ax.set_xticks(x); ax.set_xticklabels(subjects, rotation=45, ha='right', color='white')
    ax.legend(facecolor='#21262d', edgecolor='#00ff88', labelcolor='white')
    ax.grid(True, alpha=0.3, axis='y'); ax.set_facecolor('#0d1117')
    ax.tick_params(colors='white')
    for sp in ax.spines.values(): sp.set_color('white')
    fig.patch.set_facecolor('#0d1117')
    plt.tight_layout()
    p = os.path.join(output_dir, 'cognitive_load_todos_sujetos_absolutos' + OUTPUT_SUFFIX + '.png')
    plt.savefig(p, dpi=200, bbox_inches='tight', facecolor='#0d1117'); plt.close()
    print(f"Guardado: {p}")

    # Normalizado
    norm = normalize_by_baseline(all_results)
    data_norm = {pl: [] for pl in phase_labels}
    for r in norm:
        for ph, pl in zip(phases, phase_labels):
            v = r['phases_normalized'].get(ph, {}).get('normalized_ratio', np.nan)
            data_norm[pl].append(v if np.isfinite(v) else np.nan)
    fig, ax = plt.subplots(figsize=(16, 8))
    fig.suptitle('Cognitive Load Normalizado por Baseline (FIX) — Baseline=1.0', fontsize=14, fontweight='bold', color='white')
    for i, (pl, color) in enumerate(zip(phase_labels, colors)):
        off = (i - 1.5) * width
        vals = [v if not np.isnan(v) else 0 for v in data_norm[pl]]
        ax.bar(x + off, vals, width, label=pl, color=color, alpha=0.9, edgecolor='white', linewidth=0.5)
    ax.axhline(y=1.0, color='yellow', ls='--', lw=2, alpha=0.7, label='Baseline')
    ax.set_xlabel('Sujeto', color='white'); ax.set_ylabel('Ratio normalizado', color='white')
    ax.set_xticks(x); ax.set_xticklabels(subjects, rotation=45, ha='right', color='white')
    ax.legend(facecolor='#21262d', edgecolor='#00ff88', labelcolor='white')
    ax.grid(True, alpha=0.3, axis='y'); ax.set_facecolor('#0d1117')
    ax.tick_params(colors='white')
    for sp in ax.spines.values(): sp.set_color('white')
    fig.patch.set_facecolor('#0d1117')
    plt.tight_layout()
    p = os.path.join(output_dir, 'cognitive_load_todos_sujetos_normalizados' + OUTPUT_SUFFIX + '.png')
    plt.savefig(p, dpi=200, bbox_inches='tight', facecolor='#0d1117'); plt.close()
    print(f"Guardado: {p}")


def main():
    print("=" * 80)
    print("PASO 4 CORREGIDO: SR REAL en bandpower")
    print("=" * 80)
    print(f"Output: {OUTPUT_DIR}")
    print(f"Ventana: {WINDOW_SECONDS}s, paso: {STEP_SECONDS}s")
    print(f"Fase corta: < {MIN_PHASE_SECONDS}s")

    csv_files, seen = [], set()
    for cand in [Path(DATA_DIR), Path(ELECTRON_DATA_DIR)]:
        if not cand.exists():
            print(f"No existe: {cand}"); continue
        for d in sorted(cand.glob('data_*')):
            name = d.name.replace('data_', '').strip().lower()
            if name in seen: continue
            csvs = sorted(d.glob('eeg_data_*.csv'))
            if not csvs: continue
            csv_files.append(csvs[-1])  # el más reciente
            seen.add(name)

    print(f"\n{len(csv_files)} archivos CSV encontrados.")
    all_results = []
    for f in sorted(csv_files):
        try:
            all_results.append(analyze_subject_cleaned(f))
        except Exception as e:
            print(f"ERROR {f.name}: {e}")
            import traceback; traceback.print_exc()

    if not all_results:
        print("Sin resultados."); return

    df = create_final_report(all_results, OUTPUT_DIR)
    create_charts(all_results, OUTPUT_DIR)
    print("\n" + "=" * 80)
    print(f"OK. Sujetos analizados: {len(all_results)}")
    print(f"Resultados en: {OUTPUT_DIR}")
    print("=" * 80)


if __name__ == '__main__':
    main()
