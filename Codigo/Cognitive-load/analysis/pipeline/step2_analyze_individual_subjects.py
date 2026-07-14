"""
Paso 2: Análisis individual por sujeto
- Revisar cada sujeto por separado
- Visualizar señales por canal
- Verificar preprocesamiento
- Calcular métricas básicas
"""

import pandas as pd
import numpy as np
from scipy import signal, stats
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

def apply_bandpass_filter(signal_data, low_freq=1.0, high_freq=40.0, sample_rate=250):
    """Aplica filtro bandpass."""
    if len(signal_data) < 3:
        return signal_data
    
    # Si la frecuencia de muestreo es muy baja, no aplicar filtro
    if sample_rate < high_freq * 2:
        return signal_data
    
    nyquist = sample_rate / 2
    low = low_freq / nyquist
    high = high_freq / nyquist
    
    # Validar que las frecuencias normalizadas estén en el rango válido
    if low >= 1.0 or high >= 1.0 or low <= 0 or high <= 0:
        return signal_data
    
    b, a = signal.butter(4, [low, high], btype='band')
    filtered = signal.filtfilt(b, a, signal_data)
    return filtered

def apply_notch_filter(signal_data, notch_freq=60.0, sample_rate=250, Q=30.0):
    """Aplica filtro notch."""
    if len(signal_data) < 3:
        return signal_data
    
    # Si la frecuencia de muestreo es muy baja, no aplicar notch
    if sample_rate < notch_freq * 2:
        return signal_data
    
    # Normalizar frecuencia (w0 debe estar entre 0 y 1)
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
    """
    Aplica preprocesamiento completo a una señal.
    
    Args:
        channel_data: Señal a preprocesar
        sample_rate: Frecuencia de muestreo detectada
        use_default_sr: Si True, usar 250 Hz para filtrado aunque sample_rate sea diferente
    
    Returns:
        filtered_signal: Señal preprocesada
    """
    # Si la frecuencia de muestreo es muy baja, usar frecuencia por defecto para filtrado
    filter_sr = SAMPLE_RATE if (use_default_sr or sample_rate < 10) else sample_rate
    
    # 1. Notch filter (60 Hz)
    notch_filtered = apply_notch_filter(channel_data, sample_rate=filter_sr)
    
    # 2. Bandpass filter (1-40 Hz)
    filtered = apply_bandpass_filter(notch_filtered, sample_rate=filter_sr)
    
    return filtered

def calculate_basic_metrics(signal_data, label=""):
    """Calcula métricas básicas de una señal."""
    if len(signal_data) == 0:
        return None
    
    metrics = {
        'n_samples': len(signal_data),
        'mean': float(np.mean(signal_data)),
        'std': float(np.std(signal_data)),
        'min': float(np.min(signal_data)),
        'max': float(np.max(signal_data)),
        'range': float(np.max(signal_data) - np.min(signal_data)),
        'median': float(np.median(signal_data)),
        'q25': float(np.percentile(signal_data, 25)),
        'q75': float(np.percentile(signal_data, 75)),
        'iqr': float(np.percentile(signal_data, 75) - np.percentile(signal_data, 25)),
        'skewness': float(stats.skew(signal_data)) if len(signal_data) > 2 else 0.0,
        'kurtosis': float(stats.kurtosis(signal_data)) if len(signal_data) > 2 else 0.0
    }
    
    return metrics

def analyze_subject(csv_path, output_dir):
    """
    Analiza un sujeto individual.
    
    Returns:
        dict con resultados del análisis
    """
    subject_name = Path(csv_path).parent.name.replace('data_', '')
    print(f"\n{'='*80}")
    print(f"ANALISIS: {subject_name.upper()}")
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
    
    fig = plt.figure(figsize=(20, 12))
    fig.suptitle(f'Análisis Individual: {subject_name}', fontsize=16, fontweight='bold')
    
    # Grid: 2 filas (canales + métricas), columnas por fase
    gs = fig.add_gridspec(3, max(n_phases, 1), hspace=0.3, wspace=0.3)
    
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
        # Si la frecuencia de muestreo es muy baja, usar frecuencia por defecto para filtrado
        use_default_for_filtering = actual_sample_rate < 10
        if use_default_for_filtering:
            print(f"    [ADVERTENCIA] Frecuencia de muestreo muy baja ({actual_sample_rate:.1f} Hz).")
            print(f"                 Usando {SAMPLE_RATE} Hz para filtrado (datos pueden tener subsampling).")
        
        eeg_filtered = np.zeros_like(eeg_matrix)
        for ch in range(n_channels):
            eeg_filtered[:, ch] = preprocess_signal(eeg_matrix[:, ch], actual_sample_rate, 
                                                     use_default_sr=use_default_for_filtering)
        
        # Aplicar CAR
        eeg_car = apply_car(eeg_filtered, bad_channel_idx=None)
        
        # Calcular métricas por canal
        channel_metrics = {}
        for ch in range(n_channels):
            raw_metrics = calculate_basic_metrics(eeg_matrix[:, ch], f"{CHANNEL_NAMES[ch]}_raw")
            filtered_metrics = calculate_basic_metrics(eeg_car[:, ch], f"{CHANNEL_NAMES[ch]}_filtered")
            channel_metrics[ch] = {
                'raw': raw_metrics,
                'filtered': filtered_metrics
            }
        
        # Calcular bandpower para Fz y Pz
        fz_raw = eeg_matrix[:, FZ_CHANNEL]
        pz_raw = eeg_matrix[:, PZ_CHANNEL]
        fz_filtered = eeg_car[:, FZ_CHANNEL]
        pz_filtered = eeg_car[:, PZ_CHANNEL]
        
        # Usar SAMPLE_RATE (250 Hz) para cálculo de bandpower, igual que en scripts anteriores
        # Aunque los datos tengan subsampling, asumimos que fueron muestreados originalmente a 250 Hz
        theta_power_raw = calculate_bandpower(fz_raw, THETA_BAND, SAMPLE_RATE)
        alpha_power_raw = calculate_bandpower(pz_raw, ALPHA_BAND, SAMPLE_RATE)
        theta_power_filtered = calculate_bandpower(fz_filtered, THETA_BAND, SAMPLE_RATE)
        alpha_power_filtered = calculate_bandpower(pz_filtered, ALPHA_BAND, SAMPLE_RATE)
        
        ratio_raw = theta_power_raw / alpha_power_raw if alpha_power_raw > 0 else 0
        ratio_filtered = theta_power_filtered / alpha_power_filtered if alpha_power_filtered > 0 else 0
        
        # Guardar resultados
        results['phases'][label] = {
            'n_samples': n_samples,
            'duration_sec': n_samples / actual_sample_rate,
            'channel_metrics': channel_metrics,
            'theta_power_fz': {
                'raw': theta_power_raw,
                'filtered': theta_power_filtered
            },
            'alpha_power_pz': {
                'raw': alpha_power_raw,
                'filtered': alpha_power_filtered
            },
            'cognitive_load_ratio': {
                'raw': ratio_raw,
                'filtered': ratio_filtered
            }
        }
        
        # Visualización: Señales por canal (Fz y Pz)
        ax1 = fig.add_subplot(gs[0, phase_idx])
        time_axis = np.arange(len(fz_filtered)) / actual_sample_rate
        
        ax1.plot(time_axis, fz_filtered, label='Fz (filtered)', color='#00ff88', linewidth=1, alpha=0.7)
        ax1.plot(time_axis, pz_filtered, label='Pz (filtered)', color='#ff8800', linewidth=1, alpha=0.7)
        ax1.set_xlabel('Tiempo (seg)', fontsize=10)
        ax1.set_ylabel('Amplitud (μV)', fontsize=10)
        ax1.set_title(f'{label.replace("_", " ").title()}\nFz y Pz', fontsize=11, fontweight='bold')
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
        
        # Visualización: Todos los canales
        ax2 = fig.add_subplot(gs[1, phase_idx])
        for ch in range(n_channels):
            offset = ch * 50  # Offset para visualización
            ax2.plot(time_axis, eeg_car[:, ch] + offset, label=CHANNEL_NAMES[ch], linewidth=0.8, alpha=0.7)
        ax2.set_xlabel('Tiempo (seg)', fontsize=10)
        ax2.set_ylabel('Canal (con offset)', fontsize=10)
        ax2.set_title('Todos los Canales (preprocesados)', fontsize=11, fontweight='bold')
        ax2.legend(ncol=4, fontsize=7, loc='upper right')
        ax2.grid(True, alpha=0.3)
        ax2.set_facecolor('#0d1117')
        ax2.tick_params(colors='white')
        for spine in ax2.spines.values():
            spine.set_color('white')
        ax2.xaxis.label.set_color('white')
        ax2.yaxis.label.set_color('white')
        ax2.title.set_color('white')
        ax2.legend(facecolor='#21262d', edgecolor='#00ff88', labelcolor='white', fontsize=7)
        
        # Métricas básicas
        ax3 = fig.add_subplot(gs[2, phase_idx])
        ax3.axis('off')
        
        metrics_text = f"METRICAS BASICAS\n{'='*40}\n\n"
        metrics_text += f"Muestras: {n_samples:,}\n"
        metrics_text += f"Duracion: {n_samples/actual_sample_rate:.1f} seg\n\n"
        
        metrics_text += f"Fz (Theta 4-7 Hz):\n"
        metrics_text += f"  Raw: {theta_power_raw:.3f}\n"
        metrics_text += f"  Filtered: {theta_power_filtered:.3f}\n\n"
        
        metrics_text += f"Pz (Alpha 8-12 Hz):\n"
        metrics_text += f"  Raw: {alpha_power_raw:.3f}\n"
        metrics_text += f"  Filtered: {alpha_power_filtered:.3f}\n\n"
        
        metrics_text += f"Cognitive Load Ratio:\n"
        metrics_text += f"  Raw: {ratio_raw:.3f}\n"
        metrics_text += f"  Filtered: {ratio_filtered:.3f}\n\n"
        
        if filtered_metrics:
            metrics_text += f"Fz Stats (filtered):\n"
            metrics_text += f"  Mean: {filtered_metrics['mean']:.2f}\n"
            metrics_text += f"  Std: {filtered_metrics['std']:.2f}\n"
            metrics_text += f"  Range: [{filtered_metrics['min']:.2f}, {filtered_metrics['max']:.2f}]\n"
        
        ax3.text(0.05, 0.95, metrics_text, transform=ax3.transAxes,
                fontsize=9, verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='#21262d', edgecolor='#00ff88', alpha=0.8),
                color='white')
        
        phase_idx += 1
        
        # Imprimir métricas en consola
        print(f"    Theta Fz (raw): {theta_power_raw:.3f}, (filtered): {theta_power_filtered:.3f}")
        print(f"    Alpha Pz (raw): {alpha_power_raw:.3f}, (filtered): {alpha_power_filtered:.3f}")
        print(f"    Ratio (raw): {ratio_raw:.3f}, (filtered): {ratio_filtered:.3f}")
    
    # Guardar figura
    output_path = os.path.join(output_dir, f'individual_analysis_{subject_name}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='#0d1117')
    print(f"\n  Grafico guardado: {output_path}")
    plt.close()
    
    return results

def create_summary_report(all_results, output_dir):
    """Crea un reporte resumen de todos los sujetos."""
    print(f"\n{'='*80}")
    print("GENERANDO REPORTE RESUMEN")
    print(f"{'='*80}")
    
    # Crear DataFrame con métricas
    summary_data = []
    
    for result in all_results:
        subject = result['subject']
        
        for phase in LABELS_OF_INTEREST:
            if phase in result['phases']:
                phase_data = result['phases'][phase]
                summary_data.append({
                    'subject': subject,
                    'phase': phase,
                    'n_samples': phase_data['n_samples'],
                    'duration_sec': phase_data['duration_sec'],
                    'theta_power_raw': phase_data['theta_power_fz']['raw'],
                    'theta_power_filtered': phase_data['theta_power_fz']['filtered'],
                    'alpha_power_raw': phase_data['alpha_power_pz']['raw'],
                    'alpha_power_filtered': phase_data['alpha_power_pz']['filtered'],
                    'ratio_raw': phase_data['cognitive_load_ratio']['raw'],
                    'ratio_filtered': phase_data['cognitive_load_ratio']['filtered']
                })
    
    if summary_data:
        df_summary = pd.DataFrame(summary_data)
        summary_path = os.path.join(output_dir, 'individual_analysis_summary.csv')
        df_summary.to_csv(summary_path, index=False)
        print(f"Reporte guardado: {summary_path}")
        
        # Imprimir tabla resumen
        print("\nRESUMEN DE RATIOS (Filtered):")
        print("-"*80)
        print(f"{'Sujeto':<15} {'Baseline Open':<18} {'Baseline Closed':<18} {'Low Load':<15} {'High Load':<15}")
        print("-"*80)
        
        for subject in df_summary['subject'].unique():
            subject_data = df_summary[df_summary['subject'] == subject]
            ratios = {}
            for phase in LABELS_OF_INTEREST:
                phase_row = subject_data[subject_data['phase'] == phase]
                if len(phase_row) > 0:
                    ratios[phase] = phase_row.iloc[0]['ratio_filtered']
                else:
                    ratios[phase] = None
            
            ratio_open = ratios.get('baseline_eyes_open')
            ratio_closed = ratios.get('baseline_eyes_closed')
            ratio_low = ratios.get('low_cognitive_load')
            ratio_high = ratios.get('high_cognitive_load')
            
            str_open = f"{ratio_open:.3f}" if ratio_open is not None else "N/A"
            str_closed = f"{ratio_closed:.3f}" if ratio_closed is not None else "N/A"
            str_low = f"{ratio_low:.3f}" if ratio_low is not None else "N/A"
            str_high = f"{ratio_high:.3f}" if ratio_high is not None else "N/A"
            
            print(f"{subject:<15} {str_open:<18} {str_closed:<18} {str_low:<15} {str_high:<15}")

def main():
    """Función principal."""
    print("="*80)
    print("PASO 2: ANALISIS INDIVIDUAL POR SUJETO")
    print("="*80)
    print("\nAnalizando cada sujeto por separado...")
    print(f"Directorio de datos: {DATA_DIR}")
    print(f"Directorio de salida: {OUTPUT_DIR}")
    
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
            result = analyze_subject(csv_file, OUTPUT_DIR)
            all_results.append(result)
        except Exception as e:
            print(f"\nERROR analizando {csv_file.name}: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # Crear reporte resumen
    if all_results:
        create_summary_report(all_results, OUTPUT_DIR)
    
    print("\n" + "="*80)
    print("ANALISIS INDIVIDUAL COMPLETADO")
    print("="*80)
    print(f"\nTotal de sujetos analizados: {len(all_results)}")
    print(f"Resultados guardados en: {OUTPUT_DIR}")

if __name__ == '__main__':
    main()
