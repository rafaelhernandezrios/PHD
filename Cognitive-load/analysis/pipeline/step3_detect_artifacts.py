"""
Paso 3: Detección y supresión de artefactos
- Analizar cada sujeto individualmente
- Detectar picos/outliers en las señales
- Visualizar artefactos detectados
- Aplicar métodos de supresión
- Recalcular métricas después de la limpieza
"""

import pandas as pd
import numpy as np
from scipy import signal, stats
from scipy.interpolate import interp1d
from pathlib import Path
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Configuración
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, 'data', 'Data-Experimento-Rafa')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output', 'analysis_output')
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# Parámetros
SAMPLE_RATE = 250  # Hz
THETA_BAND = (4.0, 7.0)  # Hz
ALPHA_BAND = (8.0, 12.0)  # Hz
FZ_CHANNEL = 3  # Canal Fz (frontal)
PZ_CHANNEL = 6  # Canal Pz (parietal)

# Mapeo de canales
CHANNEL_NAMES = {
    0: 'Fp1', 1: 'Fp2', 2: 'F3', 3: 'Fz',
    4: 'F4', 5: 'P3', 6: 'Pz', 7: 'P4'
}

# Labels de interés
LABELS_OF_INTEREST = [
    'baseline_eyes_open',
    'baseline_eyes_closed',
    'low_cognitive_load',
    'high_cognitive_load'
]

# Parámetros de detección de artefactos
Z_SCORE_THRESHOLD = 3.0  # Umbral Z-score para detectar outliers
IQR_MULTIPLIER = 3.0  # Multiplicador IQR para detección de outliers
AMPLITUDE_THRESHOLD = 200  # uV - umbral absoluto de amplitud

def apply_bandpass_filter(signal_data, low_freq=1.0, high_freq=40.0, sample_rate=250):
    """Aplica filtro bandpass."""
    if len(signal_data) < 3:
        return signal_data
    
    if sample_rate < high_freq * 2:
        return signal_data
    
    nyquist = sample_rate / 2
    low = low_freq / nyquist
    high = high_freq / nyquist
    
    if low >= 1.0 or high >= 1.0 or low <= 0 or high <= 0:
        return signal_data
    
    b, a = signal.butter(4, [low, high], btype='band')
    filtered = signal.filtfilt(b, a, signal_data)
    return filtered

def apply_notch_filter(signal_data, notch_freq=60.0, sample_rate=250, Q=30.0):
    """Aplica filtro notch."""
    if len(signal_data) < 3:
        return signal_data
    
    if sample_rate < notch_freq * 2:
        return signal_data
    
    nyquist = sample_rate / 2
    w0 = notch_freq / nyquist
    
    if w0 >= 1.0 or w0 <= 0:
        return signal_data
    
    b, a = signal.iirnotch(w0, Q)
    filtered = signal.filtfilt(b, a, signal_data)
    return filtered

def apply_car(eeg_matrix, bad_channel_idx=None):
    """Aplica Common Average Reference (CAR)."""
    n_samples, n_channels = eeg_matrix.shape
    
    if bad_channel_idx is not None:
        valid_channels = [i for i in range(n_channels) if i != bad_channel_idx]
    else:
        valid_channels = list(range(n_channels))
    
    if len(valid_channels) == 0:
        return eeg_matrix
    
    car_reference = np.mean(eeg_matrix[:, valid_channels], axis=1, keepdims=True)
    eeg_car = eeg_matrix - car_reference
    
    return eeg_car

def detect_artifacts_zscore(signal_data, threshold=Z_SCORE_THRESHOLD):
    """
    Detecta artefactos usando Z-score.
    
    Returns:
        mask: Array booleano donde True indica artefacto
        n_artifacts: Número de artefactos detectados
    """
    if len(signal_data) < 3:
        return np.zeros(len(signal_data), dtype=bool), 0
    
    z_scores = np.abs(stats.zscore(signal_data))
    mask = z_scores > threshold
    return mask, np.sum(mask)

def detect_artifacts_iqr(signal_data, multiplier=IQR_MULTIPLIER):
    """
    Detecta artefactos usando IQR (Interquartile Range).
    
    Returns:
        mask: Array booleano donde True indica artefacto
        n_artifacts: Número de artefactos detectados
    """
    if len(signal_data) < 4:
        return np.zeros(len(signal_data), dtype=bool), 0
    
    q1 = np.percentile(signal_data, 25)
    q3 = np.percentile(signal_data, 75)
    iqr = q3 - q1
    
    lower_bound = q1 - multiplier * iqr
    upper_bound = q3 + multiplier * iqr
    
    mask = (signal_data < lower_bound) | (signal_data > upper_bound)
    return mask, np.sum(mask)

def detect_artifacts_amplitude(signal_data, threshold=AMPLITUDE_THRESHOLD):
    """
    Detecta artefactos usando umbral absoluto de amplitud.
    
    Returns:
        mask: Array booleano donde True indica artefacto
        n_artifacts: Número de artefactos detectados
    """
    mask = np.abs(signal_data) > threshold
    return mask, np.sum(mask)

def detect_artifacts_combined(signal_data, z_threshold=Z_SCORE_THRESHOLD, 
                              iqr_multiplier=IQR_MULTIPLIER, 
                              amp_threshold=AMPLITUDE_THRESHOLD):
    """
    Detecta artefactos usando combinación de métodos.
    
    Returns:
        mask: Array booleano donde True indica artefacto
        n_artifacts: Número de artefactos detectados
        method_used: Método que detectó cada artefacto
    """
    mask_z, n_z = detect_artifacts_zscore(signal_data, z_threshold)
    mask_iqr, n_iqr = detect_artifacts_iqr(signal_data, iqr_multiplier)
    mask_amp, n_amp = detect_artifacts_amplitude(signal_data, amp_threshold)
    
    # Combinar: un artefacto si es detectado por cualquiera de los métodos
    mask_combined = mask_z | mask_iqr | mask_amp
    
    # Registrar qué método detectó cada punto
    method_used = np.zeros(len(signal_data), dtype=int)
    method_used[mask_z] = 1  # Z-score
    method_used[mask_iqr] = 2  # IQR
    method_used[mask_amp] = 3  # Amplitude
    
    return mask_combined, np.sum(mask_combined), method_used

def remove_artifacts_interpolation(signal_data, artifact_mask):
    """
    Elimina artefactos mediante interpolación.
    
    Args:
        signal_data: Señal original
        artifact_mask: Máscara booleana de artefactos
    
    Returns:
        cleaned_signal: Señal con artefactos interpolados
    """
    if np.sum(artifact_mask) == 0:
        return signal_data.copy()
    
    cleaned_signal = signal_data.copy()
    
    # Obtener índices válidos (sin artefactos)
    valid_indices = np.where(~artifact_mask)[0]
    artifact_indices = np.where(artifact_mask)[0]
    
    if len(valid_indices) < 2:
        # Si hay muy pocos puntos válidos, usar clipping
        return remove_artifacts_clipping(signal_data, artifact_mask)
    
    # Interpolar valores en posiciones de artefactos
    if len(artifact_indices) > 0:
        f = interp1d(valid_indices, cleaned_signal[valid_indices], 
                    kind='linear', fill_value='extrapolate', bounds_error=False)
        cleaned_signal[artifact_indices] = f(artifact_indices)
    
    return cleaned_signal

def remove_artifacts_clipping(signal_data, artifact_mask, percentile=99):
    """
    Elimina artefactos mediante clipping (limita valores extremos).
    
    Args:
        signal_data: Señal original
        artifact_mask: Máscara booleana de artefactos
        percentile: Percentil para determinar límites
    
    Returns:
        cleaned_signal: Señal con artefactos recortados
    """
    cleaned_signal = signal_data.copy()
    
    if np.sum(artifact_mask) == 0:
        return cleaned_signal
    
    # Calcular límites basados en percentiles de datos válidos
    valid_data = signal_data[~artifact_mask]
    if len(valid_data) > 0:
        lower_bound = np.percentile(valid_data, 100 - percentile)
        upper_bound = np.percentile(valid_data, percentile)
        
        # Clipping: limitar valores extremos
        cleaned_signal[artifact_mask] = np.clip(
            cleaned_signal[artifact_mask], 
            lower_bound, 
            upper_bound
        )
    else:
        # Si no hay datos válidos, usar clipping basado en toda la señal
        lower_bound = np.percentile(signal_data, 100 - percentile)
        upper_bound = np.percentile(signal_data, percentile)
        cleaned_signal[artifact_mask] = np.clip(
            cleaned_signal[artifact_mask], 
            lower_bound, 
            upper_bound
        )
    
    return cleaned_signal

def remove_artifacts_median_filter(signal_data, artifact_mask, window_size=5):
    """
    Elimina artefactos usando filtro mediana en ventanas.
    
    Args:
        signal_data: Señal original
        artifact_mask: Máscara booleana de artefactos
        window_size: Tamaño de ventana para filtro mediana
    
    Returns:
        cleaned_signal: Señal con artefactos filtrados
    """
    cleaned_signal = signal_data.copy()
    
    if np.sum(artifact_mask) == 0:
        return cleaned_signal
    
    # Aplicar filtro mediana solo en regiones con artefactos
    artifact_indices = np.where(artifact_mask)[0]
    
    for idx in artifact_indices:
        start = max(0, idx - window_size // 2)
        end = min(len(signal_data), idx + window_size // 2 + 1)
        window = signal_data[start:end]
        cleaned_signal[idx] = np.median(window)
    
    return cleaned_signal

def calculate_bandpower(signal_data, freq_band, sample_rate):
    """Calcula la potencia espectral en una banda usando Welch."""
    if len(signal_data) < sample_rate // 2:
        return 0.0
    
    nperseg = min(len(signal_data), sample_rate)
    freqs, psd = signal.welch(signal_data, sample_rate, nperseg=nperseg, noverlap=nperseg//2)
    
    idx_band = np.logical_and(freqs >= freq_band[0], freqs <= freq_band[1])
    
    if np.sum(idx_band) == 0:
        return 0.0
    
    bandpower = np.trapz(psd[idx_band], freqs[idx_band])
    return bandpower

def preprocess_signal(channel_data, sample_rate=250, use_default_sr=False):
    """Aplica preprocesamiento completo a una señal."""
    filter_sr = SAMPLE_RATE if (use_default_sr or sample_rate < 10) else sample_rate
    
    notch_filtered = apply_notch_filter(channel_data, sample_rate=filter_sr)
    filtered = apply_bandpass_filter(notch_filtered, sample_rate=filter_sr)
    
    return filtered

def analyze_artifacts_subject(csv_path, output_dir):
    """
    Analiza artefactos en un sujeto individual.
    
    Returns:
        dict con resultados del análisis
    """
    subject_name = Path(csv_path).parent.name.replace('data_', '')
    print(f"\n{'='*80}")
    print(f"ANALISIS DE ARTEFACTOS: {subject_name.upper()}")
    print(f"{'='*80}")
    
    # Cargar datos
    df = pd.read_csv(csv_path)
    print(f"Total de muestras: {len(df):,}")
    
    # Calcular frecuencia de muestreo real
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')
        df_sorted = df.sort_values('timestamp')
        time_diffs = df_sorted['timestamp'].diff().dropna()
        median_interval = time_diffs.median()
        actual_sample_rate = 1.0 / median_interval if median_interval > 0 else SAMPLE_RATE
        print(f"Frecuencia de muestreo detectada: {actual_sample_rate:.1f} Hz")
    else:
        actual_sample_rate = SAMPLE_RATE
        print(f"Usando frecuencia por defecto: {SAMPLE_RATE} Hz")
    
    use_default_for_filtering = actual_sample_rate < 10
    
    # Análisis por fase
    results = {
        'subject': subject_name,
        'total_samples': len(df),
        'sample_rate': actual_sample_rate,
        'phases': {}
    }
    
    # Crear visualización
    n_phases = len([l for l in LABELS_OF_INTEREST if l in df['label'].values])
    if n_phases == 0:
        print("  Advertencia: No se encontraron fases de interes")
        return results
    
    fig = plt.figure(figsize=(24, 16))
    fig.suptitle(f'Detección de Artefactos: {subject_name}', fontsize=16, fontweight='bold')
    
    # Grid: 4 filas (raw, artifacts, cleaned, metrics), columnas por fase
    gs = fig.add_gridspec(4, max(n_phases, 1), hspace=0.4, wspace=0.3)
    
    phase_idx = 0
    for label in LABELS_OF_INTEREST:
        phase_data = df[df['label'] == label].copy()
        
        if len(phase_data) == 0:
            continue
        
        print(f"\n  Fase: {label}")
        print(f"    Muestras: {len(phase_data):,}")
        
        # Ordenar por timestamp si existe
        if 'timestamp' in phase_data.columns:
            phase_data = phase_data.sort_values('timestamp')
        
        # Construir matriz EEG
        n_samples = len(phase_data)
        n_channels = 8
        eeg_matrix = np.zeros((n_samples, n_channels))
        
        for ch in range(n_channels):
            eeg_matrix[:, ch] = phase_data[f'channel_{ch}'].values
        
        # Preprocesar
        eeg_filtered = np.zeros_like(eeg_matrix)
        for ch in range(n_channels):
            eeg_filtered[:, ch] = preprocess_signal(
                eeg_matrix[:, ch], actual_sample_rate, 
                use_default_sr=use_default_for_filtering
            )
        
        # Aplicar CAR
        eeg_car = apply_car(eeg_filtered, bad_channel_idx=None)
        
        # Analizar artefactos en Fz y Pz
        fz_signal = eeg_car[:, FZ_CHANNEL]
        pz_signal = eeg_car[:, PZ_CHANNEL]
        
        # Detectar artefactos
        fz_artifacts, fz_n_artifacts, fz_methods = detect_artifacts_combined(fz_signal)
        pz_artifacts, pz_n_artifacts, pz_methods = detect_artifacts_combined(pz_signal)
        
        print(f"    Artefactos Fz: {fz_n_artifacts} ({100*fz_n_artifacts/len(fz_signal):.1f}%)")
        print(f"    Artefactos Pz: {pz_n_artifacts} ({100*pz_n_artifacts/len(pz_signal):.1f}%)")
        
        # Limpiar artefactos
        fz_cleaned = remove_artifacts_interpolation(fz_signal, fz_artifacts)
        pz_cleaned = remove_artifacts_interpolation(pz_signal, pz_artifacts)
        
        # Calcular métricas antes y después
        time_axis = np.arange(len(fz_signal)) / SAMPLE_RATE
        
        # Bandpower antes y después
        theta_before = calculate_bandpower(fz_signal, THETA_BAND, SAMPLE_RATE)
        alpha_before = calculate_bandpower(pz_signal, ALPHA_BAND, SAMPLE_RATE)
        ratio_before = theta_before / alpha_before if alpha_before > 0 else 0
        
        theta_after = calculate_bandpower(fz_cleaned, THETA_BAND, SAMPLE_RATE)
        alpha_after = calculate_bandpower(pz_cleaned, ALPHA_BAND, SAMPLE_RATE)
        ratio_after = theta_after / alpha_after if alpha_after > 0 else 0
        
        # Guardar resultados
        results['phases'][label] = {
            'n_samples': n_samples,
            'fz_artifacts': int(fz_n_artifacts),
            'pz_artifacts': int(pz_n_artifacts),
            'fz_artifact_pct': float(100*fz_n_artifacts/len(fz_signal)),
            'pz_artifact_pct': float(100*pz_n_artifacts/len(pz_signal)),
            'ratio_before': float(ratio_before),
            'ratio_after': float(ratio_after),
            'theta_before': float(theta_before),
            'theta_after': float(theta_after),
            'alpha_before': float(alpha_before),
            'alpha_after': float(alpha_after)
        }
        
        # Visualización: Señal original
        ax1 = fig.add_subplot(gs[0, phase_idx])
        ax1.plot(time_axis, fz_signal, label='Fz', color='#00ff88', linewidth=1, alpha=0.7)
        ax1.plot(time_axis, pz_signal, label='Pz', color='#ff8800', linewidth=1, alpha=0.7)
        ax1.set_xlabel('Tiempo (seg)', fontsize=9)
        ax1.set_ylabel('Amplitud (uV)', fontsize=9)
        ax1.set_title(f'{label.replace("_", " ").title()}\nSeñal Original', fontsize=10, fontweight='bold')
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)
        ax1.set_facecolor('#0d1117')
        ax1.tick_params(colors='white')
        for spine in ax1.spines.values():
            spine.set_color('white')
        ax1.xaxis.label.set_color('white')
        ax1.yaxis.label.set_color('white')
        ax1.title.set_color('white')
        ax1.legend(facecolor='#21262d', edgecolor='#00ff88', labelcolor='white', fontsize=8)
        
        # Visualización: Artefactos detectados
        ax2 = fig.add_subplot(gs[1, phase_idx])
        ax2.plot(time_axis, fz_signal, color='#00ff88', linewidth=1, alpha=0.3, label='Fz')
        ax2.plot(time_axis, pz_signal, color='#ff8800', linewidth=1, alpha=0.3, label='Pz')
        
        # Marcar artefactos
        fz_artifact_indices = np.where(fz_artifacts)[0]
        pz_artifact_indices = np.where(pz_artifacts)[0]
        
        if len(fz_artifact_indices) > 0:
            ax2.scatter(time_axis[fz_artifact_indices], fz_signal[fz_artifact_indices], 
                       color='red', s=20, alpha=0.8, marker='x', label=f'Fz artifacts ({fz_n_artifacts})')
        if len(pz_artifact_indices) > 0:
            ax2.scatter(time_axis[pz_artifact_indices], pz_signal[pz_artifact_indices], 
                       color='magenta', s=20, alpha=0.8, marker='x', label=f'Pz artifacts ({pz_n_artifacts})')
        
        ax2.set_xlabel('Tiempo (seg)', fontsize=9)
        ax2.set_ylabel('Amplitud (uV)', fontsize=9)
        ax2.set_title('Artefactos Detectados', fontsize=10, fontweight='bold')
        ax2.legend(fontsize=7)
        ax2.grid(True, alpha=0.3)
        ax2.set_facecolor('#0d1117')
        ax2.tick_params(colors='white')
        for spine in ax2.spines.values():
            spine.set_color('white')
        ax2.xaxis.label.set_color('white')
        ax2.yaxis.label.set_color('white')
        ax2.title.set_color('white')
        ax2.legend(facecolor='#21262d', edgecolor='#00ff88', labelcolor='white', fontsize=7)
        
        # Visualización: Señal limpiada
        ax3 = fig.add_subplot(gs[2, phase_idx])
        ax3.plot(time_axis, fz_cleaned, label='Fz (cleaned)', color='#00ff88', linewidth=1, alpha=0.7)
        ax3.plot(time_axis, pz_cleaned, label='Pz (cleaned)', color='#ff8800', linewidth=1, alpha=0.7)
        ax3.set_xlabel('Tiempo (seg)', fontsize=9)
        ax3.set_ylabel('Amplitud (uV)', fontsize=9)
        ax3.set_title('Señal Limpiada', fontsize=10, fontweight='bold')
        ax3.legend(fontsize=8)
        ax3.grid(True, alpha=0.3)
        ax3.set_facecolor('#0d1117')
        ax3.tick_params(colors='white')
        for spine in ax3.spines.values():
            spine.set_color('white')
        ax3.xaxis.label.set_color('white')
        ax3.yaxis.label.set_color('white')
        ax3.title.set_color('white')
        ax3.legend(facecolor='#21262d', edgecolor='#00ff88', labelcolor='white', fontsize=8)
        
        # Métricas
        ax4 = fig.add_subplot(gs[3, phase_idx])
        ax4.axis('off')
        
        metrics_text = f"METRICAS\n{'='*40}\n\n"
        metrics_text += f"Artefactos:\n"
        metrics_text += f"  Fz: {fz_n_artifacts} ({100*fz_n_artifacts/len(fz_signal):.1f}%)\n"
        metrics_text += f"  Pz: {pz_n_artifacts} ({100*pz_n_artifacts/len(pz_signal):.1f}%)\n\n"
        
        metrics_text += f"Cognitive Load Ratio:\n"
        metrics_text += f"  Antes: {ratio_before:.3f}\n"
        metrics_text += f"  Despues: {ratio_after:.3f}\n"
        metrics_text += f"  Cambio: {((ratio_after/ratio_before - 1)*100) if ratio_before > 0 else 0:.1f}%\n\n"
        
        metrics_text += f"Theta Fz:\n"
        metrics_text += f"  Antes: {theta_before:.3f}\n"
        metrics_text += f"  Despues: {theta_after:.3f}\n\n"
        
        metrics_text += f"Alpha Pz:\n"
        metrics_text += f"  Antes: {alpha_before:.3f}\n"
        metrics_text += f"  Despues: {alpha_after:.3f}\n"
        
        ax4.text(0.05, 0.95, metrics_text, transform=ax4.transAxes,
                fontsize=9, verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='#21262d', edgecolor='#00ff88', alpha=0.8),
                color='white')
        
        phase_idx += 1
        
        print(f"    Ratio antes: {ratio_before:.3f}, despues: {ratio_after:.3f}")
    
    # Guardar figura
    output_path = os.path.join(output_dir, f'artifact_analysis_{subject_name}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='#0d1117')
    print(f"\n  Grafico guardado: {output_path}")
    plt.close()
    
    return results

def create_artifacts_summary(all_results, output_dir):
    """Crea un reporte resumen de artefactos."""
    print(f"\n{'='*80}")
    print("GENERANDO REPORTE RESUMEN DE ARTEFACTOS")
    print(f"{'='*80}")
    
    summary_data = []
    
    for result in all_results:
        subject = result['subject']
        
        for phase in LABELS_OF_INTEREST:
            if phase in result['phases']:
                phase_data = result['phases'][phase]
                summary_data.append({
                    'subject': subject,
                    'phase': phase,
                    'fz_artifacts': phase_data['fz_artifacts'],
                    'pz_artifacts': phase_data['pz_artifacts'],
                    'fz_artifact_pct': phase_data['fz_artifact_pct'],
                    'pz_artifact_pct': phase_data['pz_artifact_pct'],
                    'ratio_before': phase_data['ratio_before'],
                    'ratio_after': phase_data['ratio_after'],
                    'ratio_change_pct': ((phase_data['ratio_after'] / phase_data['ratio_before'] - 1) * 100) 
                                        if phase_data['ratio_before'] > 0 else 0
                })
    
    if summary_data:
        df_summary = pd.DataFrame(summary_data)
        summary_path = os.path.join(output_dir, 'artifact_analysis_summary.csv')
        df_summary.to_csv(summary_path, index=False)
        print(f"Reporte guardado: {summary_path}")
        
        # Imprimir tabla resumen
        print("\nRESUMEN DE ARTEFACTOS:")
        print("-"*100)
        print(f"{'Sujeto':<12} {'Fase':<20} {'Fz Art%':<10} {'Pz Art%':<10} {'Ratio Antes':<12} {'Ratio Despues':<14} {'Cambio%':<10}")
        print("-"*100)
        
        for _, row in df_summary.iterrows():
            print(f"{row['subject']:<12} {row['phase']:<20} {row['fz_artifact_pct']:<10.1f} "
                  f"{row['pz_artifact_pct']:<10.1f} {row['ratio_before']:<12.3f} "
                  f"{row['ratio_after']:<14.3f} {row['ratio_change_pct']:<10.1f}")

def main():
    """Función principal."""
    print("="*80)
    print("PASO 3: DETECCION Y SUPRESION DE ARTEFACTOS")
    print("="*80)
    print("\nAnalizando artefactos en cada sujeto...")
    print(f"Directorio de datos: {DATA_DIR}")
    print(f"Directorio de salida: {OUTPUT_DIR}")
    print(f"\nParametros de deteccion:")
    print(f"  Z-score threshold: {Z_SCORE_THRESHOLD}")
    print(f"  IQR multiplier: {IQR_MULTIPLIER}")
    print(f"  Amplitude threshold: {AMPLITUDE_THRESHOLD} uV")
    
    # Buscar archivos CSV
    csv_files = []
    data_path = Path(DATA_DIR)
    
    if data_path.exists():
        for data_dir in data_path.glob('data_*'):
            csv_files.extend(data_dir.glob('eeg_data_*.csv'))
    else:
        print(f"ERROR: No se encontro el directorio {DATA_DIR}")
        return
    
    if not csv_files:
        print("No se encontraron archivos CSV")
        return
    
    print(f"\nEncontrados {len(csv_files)} archivos CSV")
    
    # Analizar cada sujeto
    all_results = []
    for csv_file in sorted(csv_files):
        try:
            result = analyze_artifacts_subject(csv_file, OUTPUT_DIR)
            all_results.append(result)
        except Exception as e:
            print(f"\nERROR analizando {csv_file.name}: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # Crear reporte resumen
    if all_results:
        create_artifacts_summary(all_results, OUTPUT_DIR)
    
    print("\n" + "="*80)
    print("ANALISIS DE ARTEFACTOS COMPLETADO")
    print("="*80)
    print(f"\nTotal de sujetos analizados: {len(all_results)}")
    print(f"Resultados guardados en: {OUTPUT_DIR}")

if __name__ == '__main__':
    main()
