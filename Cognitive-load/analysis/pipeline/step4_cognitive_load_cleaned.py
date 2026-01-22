"""
Paso 4: Cálculo de Cognitive Load con datos limpiados
- Aplica limpieza de artefactos a todos los datos
- Recalcula ratios de cognitive load
- Compara resultados antes/después de la limpieza
- Genera visualizaciones y reportes finales
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
WINDOW_SAMPLES = 250  # Ventana de 1 segundo a 250 Hz

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
Z_SCORE_THRESHOLD = 3.0
IQR_MULTIPLIER = 3.0
AMPLITUDE_THRESHOLD = 200  # uV

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

def detect_artifacts_combined(signal_data, z_threshold=Z_SCORE_THRESHOLD, 
                              iqr_multiplier=IQR_MULTIPLIER, 
                              amp_threshold=AMPLITUDE_THRESHOLD):
    """Detecta artefactos usando combinación de métodos."""
    if len(signal_data) < 3:
        return np.zeros(len(signal_data), dtype=bool), 0
    
    # Z-score
    z_scores = np.abs(stats.zscore(signal_data))
    mask_z = z_scores > z_threshold
    
    # IQR
    if len(signal_data) >= 4:
        q1 = np.percentile(signal_data, 25)
        q3 = np.percentile(signal_data, 75)
        iqr = q3 - q1
        lower_bound = q1 - iqr_multiplier * iqr
        upper_bound = q3 + iqr_multiplier * iqr
        mask_iqr = (signal_data < lower_bound) | (signal_data > upper_bound)
    else:
        mask_iqr = np.zeros(len(signal_data), dtype=bool)
    
    # Amplitude
    mask_amp = np.abs(signal_data) > amp_threshold
    
    # Combinar
    mask_combined = mask_z | mask_iqr | mask_amp
    return mask_combined, np.sum(mask_combined)

def remove_artifacts_interpolation(signal_data, artifact_mask):
    """Elimina artefactos mediante interpolación."""
    if np.sum(artifact_mask) == 0:
        return signal_data.copy()
    
    cleaned_signal = signal_data.copy()
    valid_indices = np.where(~artifact_mask)[0]
    artifact_indices = np.where(artifact_mask)[0]
    
    if len(valid_indices) < 2:
        # Si hay muy pocos puntos válidos, usar clipping
        valid_data = signal_data[~artifact_mask]
        if len(valid_data) > 0:
            lower_bound = np.percentile(valid_data, 1)
            upper_bound = np.percentile(valid_data, 99)
            cleaned_signal[artifact_indices] = np.clip(
                cleaned_signal[artifact_indices], lower_bound, upper_bound
            )
        return cleaned_signal
    
    # Interpolar
    if len(artifact_indices) > 0:
        f = interp1d(valid_indices, cleaned_signal[valid_indices], 
                    kind='linear', fill_value='extrapolate', bounds_error=False)
        cleaned_signal[artifact_indices] = f(artifact_indices)
    
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

def calculate_cognitive_load_by_phase(fz_signal, pz_signal, sample_rate=250):
    """
    Calcula cognitive load usando ventaneo deslizante.
    
    Returns:
        ratios: Lista de ratios calculados
        theta_powers: Lista de potencias Theta
        alpha_powers: Lista de potencias Alpha
    """
    ratios = []
    theta_powers = []
    alpha_powers = []
    
    min_samples = sample_rate // 2
    if len(fz_signal) < min_samples or len(pz_signal) < min_samples:
        return ratios, theta_powers, alpha_powers
    
    # Ajustar tamaño de ventana según datos disponibles
    if len(fz_signal) >= WINDOW_SAMPLES:
        actual_window = WINDOW_SAMPLES
        step_samples = WINDOW_SAMPLES // 2
    else:
        actual_window = len(fz_signal)
        step_samples = actual_window
    
    # Ventaneo deslizante
    for start_idx in range(0, len(fz_signal) - actual_window + 1, step_samples):
        end_idx = start_idx + actual_window
        fz_window = fz_signal[start_idx:end_idx]
        pz_window = pz_signal[start_idx:end_idx]
        
        theta_power = calculate_bandpower(fz_window, THETA_BAND, sample_rate)
        alpha_power = calculate_bandpower(pz_window, ALPHA_BAND, sample_rate)
        
        if alpha_power > 0 and np.isfinite(theta_power) and np.isfinite(alpha_power):
            ratio = theta_power / alpha_power
            if np.isfinite(ratio) and 0.01 < ratio < 100:  # Sanity filter
                ratios.append(ratio)
                theta_powers.append(theta_power)
                alpha_powers.append(alpha_power)
    
    return ratios, theta_powers, alpha_powers

def analyze_subject_cleaned(csv_path, output_dir):
    """
    Analiza un sujeto con datos limpiados y calcula cognitive load.
    
    Returns:
        dict con resultados
    """
    subject_name = Path(csv_path).parent.name.replace('data_', '')
    print(f"\n{'='*80}")
    print(f"ANALISIS CON DATOS LIMPIADOS: {subject_name.upper()}")
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
    else:
        actual_sample_rate = SAMPLE_RATE
    
    use_default_for_filtering = actual_sample_rate < 10
    
    results = {
        'subject': subject_name,
        'phases': {}
    }
    
    for label in LABELS_OF_INTEREST:
        phase_data = df[df['label'] == label].copy()
        
        if len(phase_data) == 0:
            continue
        
        print(f"\n  Fase: {label}")
        print(f"    Muestras: {len(phase_data):,}")
        
        # Ordenar por timestamp
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
        
        # Extraer Fz y Pz
        fz_signal = eeg_car[:, FZ_CHANNEL]
        pz_signal = eeg_car[:, PZ_CHANNEL]
        
        # Detectar y limpiar artefactos
        fz_artifacts, fz_n_artifacts = detect_artifacts_combined(fz_signal)
        pz_artifacts, pz_n_artifacts = detect_artifacts_combined(pz_signal)
        
        fz_cleaned = remove_artifacts_interpolation(fz_signal, fz_artifacts)
        pz_cleaned = remove_artifacts_interpolation(pz_signal, pz_artifacts)
        
        print(f"    Artefactos Fz: {fz_n_artifacts} ({100*fz_n_artifacts/len(fz_signal):.1f}%)")
        print(f"    Artefactos Pz: {pz_n_artifacts} ({100*pz_n_artifacts/len(pz_signal):.1f}%)")
        
        # Calcular cognitive load ANTES de limpiar
        ratios_before, theta_before, alpha_before = calculate_cognitive_load_by_phase(
            fz_signal, pz_signal, SAMPLE_RATE
        )
        
        # Calcular cognitive load DESPUÉS de limpiar
        ratios_after, theta_after, alpha_after = calculate_cognitive_load_by_phase(
            fz_cleaned, pz_cleaned, SAMPLE_RATE
        )
        
        # Estadísticas
        mean_ratio_before = np.mean(ratios_before) if ratios_before else 0
        mean_ratio_after = np.mean(ratios_after) if ratios_after else 0
        median_ratio_before = np.median(ratios_before) if ratios_before else 0
        median_ratio_after = np.median(ratios_after) if ratios_after else 0
        std_ratio_before = np.std(ratios_before) if ratios_before else 0
        std_ratio_after = np.std(ratios_after) if ratios_after else 0
        
        results['phases'][label] = {
            'n_samples': n_samples,
            'fz_artifacts': int(fz_n_artifacts),
            'pz_artifacts': int(pz_n_artifacts),
            'fz_artifact_pct': float(100*fz_n_artifacts/len(fz_signal)),
            'pz_artifact_pct': float(100*pz_n_artifacts/len(pz_signal)),
            'n_windows_before': len(ratios_before),
            'n_windows_after': len(ratios_after),
            'mean_ratio_before': float(mean_ratio_before),
            'mean_ratio_after': float(mean_ratio_after),
            'median_ratio_before': float(median_ratio_before),
            'median_ratio_after': float(median_ratio_after),
            'std_ratio_before': float(std_ratio_before),
            'std_ratio_after': float(std_ratio_after),
            'ratios_before': ratios_before,
            'ratios_after': ratios_after
        }
        
        print(f"    Ventanas validas antes: {len(ratios_before)}")
        print(f"    Ventanas validas despues: {len(ratios_after)}")
        print(f"    Ratio medio antes: {mean_ratio_before:.3f}")
        print(f"    Ratio medio despues: {mean_ratio_after:.3f}")
    
    return results

def create_comparison_visualization(all_results, output_dir):
    """Crea visualización comparativa de todos los sujetos."""
    print(f"\n{'='*80}")
    print("GENERANDO VISUALIZACION COMPARATIVA")
    print(f"{'='*80}")
    
    # Preparar datos para gráficos
    comparison_data = []
    
    for result in all_results:
        subject = result['subject']
        for phase in LABELS_OF_INTEREST:
            if phase in result['phases']:
                phase_data = result['phases'][phase]
                comparison_data.append({
                    'subject': subject,
                    'phase': phase,
                    'mean_ratio_before': phase_data['mean_ratio_before'],
                    'mean_ratio_after': phase_data['mean_ratio_after'],
                    'median_ratio_before': phase_data['median_ratio_before'],
                    'median_ratio_after': phase_data['median_ratio_after']
                })
    
    if not comparison_data:
        print("No hay datos para visualizar")
        return
    
    df_comp = pd.DataFrame(comparison_data)
    
    # Crear figura con múltiples subplots
    fig = plt.figure(figsize=(20, 12))
    fig.suptitle('Comparacion Cognitive Load: Antes vs Despues de Limpieza', 
                 fontsize=16, fontweight='bold')
    
    # Subplot 1: Boxplot comparativo por fase
    ax1 = plt.subplot(2, 2, 1)
    
    phases_to_plot = ['low_cognitive_load', 'high_cognitive_load']
    before_data = []
    after_data = []
    labels = []
    
    for phase in phases_to_plot:
        phase_before = []
        phase_after = []
        for result in all_results:
            if phase in result['phases']:
                ratios_before = result['phases'][phase]['ratios_before']
                ratios_after = result['phases'][phase]['ratios_after']
                phase_before.extend(ratios_before)
                phase_after.extend(ratios_after)
        
        if phase_before and phase_after:
            before_data.append(phase_before)
            after_data.append(phase_after)
            labels.append(phase.replace('_', ' ').title())
    
    if before_data and after_data:
        positions = np.arange(len(labels))
        width = 0.35
        
        bp1 = ax1.boxplot(before_data, positions=positions - width/2, widths=width, 
                         patch_artist=True, labels=labels)
        bp2 = ax1.boxplot(after_data, positions=positions + width/2, widths=width, 
                         patch_artist=True)
        
        for patch in bp1['boxes']:
            patch.set_facecolor('#ff8800')
            patch.set_alpha(0.7)
        for patch in bp2['boxes']:
            patch.set_facecolor('#00ff88')
            patch.set_alpha(0.7)
        
        ax1.set_ylabel('Cognitive Load Ratio', fontsize=11)
        ax1.set_title('Distribucion de Ratios por Fase', fontsize=12, fontweight='bold')
        ax1.legend([bp1['boxes'][0], bp2['boxes'][0]], ['Antes', 'Despues'], fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.set_facecolor('#0d1117')
        ax1.tick_params(colors='white')
        for spine in ax1.spines.values():
            spine.set_color('white')
        ax1.xaxis.label.set_color('white')
        ax1.yaxis.label.set_color('white')
        ax1.title.set_color('white')
        ax1.legend(facecolor='#21262d', edgecolor='#00ff88', labelcolor='white', fontsize=10)
    
    # Subplot 2: Comparación de medias por sujeto
    ax2 = plt.subplot(2, 2, 2)
    
    subjects = df_comp['subject'].unique()
    low_before = []
    low_after = []
    high_before = []
    high_after = []
    
    for subject in subjects:
        subject_data = df_comp[df_comp['subject'] == subject]
        low_data = subject_data[subject_data['phase'] == 'low_cognitive_load']
        high_data = subject_data[subject_data['phase'] == 'high_cognitive_load']
        
        if len(low_data) > 0:
            low_before.append(low_data.iloc[0]['mean_ratio_before'])
            low_after.append(low_data.iloc[0]['mean_ratio_after'])
        else:
            low_before.append(0)
            low_after.append(0)
        
        if len(high_data) > 0:
            high_before.append(high_data.iloc[0]['mean_ratio_before'])
            high_after.append(high_data.iloc[0]['mean_ratio_after'])
        else:
            high_before.append(0)
            high_after.append(0)
    
    x = np.arange(len(subjects))
    width = 0.35
    
    ax2.bar(x - width/2, low_before, width, label='Low (antes)', color='#ff8800', alpha=0.7)
    ax2.bar(x - width/2 + width, low_after, width, label='Low (despues)', color='#00ff88', alpha=0.7)
    ax2.bar(x + width/2, high_before, width, label='High (antes)', color='#ff4444', alpha=0.7)
    ax2.bar(x + width/2 + width, high_after, width, label='High (despues)', color='#44ff44', alpha=0.7)
    
    ax2.set_xlabel('Sujeto', fontsize=11)
    ax2.set_ylabel('Mean Ratio', fontsize=11)
    ax2.set_title('Comparacion por Sujeto', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(subjects, rotation=45, ha='right')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_facecolor('#0d1117')
    ax2.tick_params(colors='white')
    for spine in ax2.spines.values():
        spine.set_color('white')
    ax2.xaxis.label.set_color('white')
    ax2.yaxis.label.set_color('white')
    ax2.title.set_color('white')
    ax2.legend(facecolor='#21262d', edgecolor='#00ff88', labelcolor='white', fontsize=9)
    
    # Subplot 3: Scatter plot antes vs después
    ax3 = plt.subplot(2, 2, 3)
    
    low_before_list = df_comp[df_comp['phase'] == 'low_cognitive_load']['mean_ratio_before'].values
    low_after_list = df_comp[df_comp['phase'] == 'low_cognitive_load']['mean_ratio_after'].values
    high_before_list = df_comp[df_comp['phase'] == 'high_cognitive_load']['mean_ratio_before'].values
    high_after_list = df_comp[df_comp['phase'] == 'high_cognitive_load']['mean_ratio_after'].values
    
    ax3.scatter(low_before_list, low_after_list, color='#00ff88', alpha=0.7, s=100, 
               label='Low Load', edgecolors='white', linewidths=1)
    ax3.scatter(high_before_list, high_after_list, color='#ff8800', alpha=0.7, s=100, 
               label='High Load', edgecolors='white', linewidths=1)
    
    # Línea de identidad
    max_val = max(max(low_before_list, default=[0]), max(low_after_list, default=[0]),
                  max(high_before_list, default=[0]), max(high_after_list, default=[0]))
    ax3.plot([0, max_val], [0, max_val], 'r--', alpha=0.5, label='Identidad')
    
    ax3.set_xlabel('Ratio Antes', fontsize=11)
    ax3.set_ylabel('Ratio Despues', fontsize=11)
    ax3.set_title('Antes vs Despues de Limpieza', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.set_facecolor('#0d1117')
    ax3.tick_params(colors='white')
    for spine in ax3.spines.values():
        spine.set_color('white')
    ax3.xaxis.label.set_color('white')
    ax3.yaxis.label.set_color('white')
    ax3.title.set_color('white')
    ax3.legend(facecolor='#21262d', edgecolor='#00ff88', labelcolor='white', fontsize=10)
    
    # Subplot 4: Tabla resumen
    ax4 = plt.subplot(2, 2, 4)
    ax4.axis('off')
    
    # Crear tabla resumen
    summary_text = "RESUMEN DE HIPOTESIS\n" + "="*50 + "\n\n"
    summary_text += "Hipotesis: High Load > Low Load\n\n"
    
    subjects_meeting_hypothesis_before = 0
    subjects_meeting_hypothesis_after = 0
    
    for subject in subjects:
        subject_data = df_comp[df_comp['subject'] == subject]
        low_data = subject_data[subject_data['phase'] == 'low_cognitive_load']
        high_data = subject_data[subject_data['phase'] == 'high_cognitive_load']
        
        if len(low_data) > 0 and len(high_data) > 0:
            low_before_val = low_data.iloc[0]['mean_ratio_before']
            high_before_val = high_data.iloc[0]['mean_ratio_before']
            low_after_val = low_data.iloc[0]['mean_ratio_after']
            high_after_val = high_data.iloc[0]['mean_ratio_after']
            
            meets_before = high_before_val > low_before_val
            meets_after = high_after_val > low_after_val
            
            if meets_before:
                subjects_meeting_hypothesis_before += 1
            if meets_after:
                subjects_meeting_hypothesis_after += 1
            
            status_before = "[OK]" if meets_before else "[X]"
            status_after = "[OK]" if meets_after else "[X]"
            
            summary_text += f"{subject:<12} {status_before} {status_after}\n"
    
    summary_text += f"\nTotal cumpliendo hipotesis:\n"
    summary_text += f"  Antes: {subjects_meeting_hypothesis_before}/{len(subjects)}\n"
    summary_text += f"  Despues: {subjects_meeting_hypothesis_after}/{len(subjects)}\n"
    
    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes,
            fontsize=10, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='#21262d', edgecolor='#00ff88', alpha=0.8),
            color='white')
    
    plt.tight_layout()
    
    # Guardar figura
    output_path = os.path.join(output_dir, 'cognitive_load_comparison_cleaned.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='#0d1117')
    print(f"Grafico guardado: {output_path}")
    plt.close()

def create_final_report(all_results, output_dir):
    """Crea reporte final con ratios limpiados."""
    print(f"\n{'='*80}")
    print("GENERANDO REPORTE FINAL")
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
                    'n_samples': phase_data['n_samples'],
                    'fz_artifacts': phase_data['fz_artifacts'],
                    'pz_artifacts': phase_data['pz_artifacts'],
                    'fz_artifact_pct': phase_data['fz_artifact_pct'],
                    'pz_artifact_pct': phase_data['pz_artifact_pct'],
                    'n_windows_before': phase_data['n_windows_before'],
                    'n_windows_after': phase_data['n_windows_after'],
                    'mean_ratio_before': phase_data['mean_ratio_before'],
                    'mean_ratio_after': phase_data['mean_ratio_after'],
                    'median_ratio_before': phase_data['median_ratio_before'],
                    'median_ratio_after': phase_data['median_ratio_after'],
                    'std_ratio_before': phase_data['std_ratio_before'],
                    'std_ratio_after': phase_data['std_ratio_after']
                })
    
    if summary_data:
        df_summary = pd.DataFrame(summary_data)
        summary_path = os.path.join(output_dir, 'cognitive_load_cleaned_summary.csv')
        df_summary.to_csv(summary_path, index=False)
        print(f"Reporte guardado: {summary_path}")
        
        # Imprimir tabla resumen
        print("\nRESUMEN DE RATIOS (DESPUES DE LIMPIEZA):")
        print("-"*100)
        print(f"{'Sujeto':<12} {'Baseline Open':<18} {'Baseline Closed':<18} {'Low Load':<15} {'High Load':<15}")
        print("-"*100)
        
        for subject in df_summary['subject'].unique():
            subject_data = df_summary[df_summary['subject'] == subject]
            ratios = {}
            for phase in LABELS_OF_INTEREST:
                phase_row = subject_data[subject_data['phase'] == phase]
                if len(phase_row) > 0:
                    ratios[phase] = phase_row.iloc[0]['mean_ratio_after']
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
            
            print(f"{subject:<12} {str_open:<18} {str_closed:<18} {str_low:<15} {str_high:<15}")
        
        # Verificar hipótesis
        print("\nVERIFICACION DE HIPOTESIS (High Load > Low Load):")
        print("-"*60)
        subjects_meeting = 0
        for subject in df_summary['subject'].unique():
            subject_data = df_summary[df_summary['subject'] == subject]
            low_data = subject_data[subject_data['phase'] == 'low_cognitive_load']
            high_data = subject_data[subject_data['phase'] == 'high_cognitive_load']
            
            if len(low_data) > 0 and len(high_data) > 0:
                low_ratio = low_data.iloc[0]['mean_ratio_after']
                high_ratio = high_data.iloc[0]['mean_ratio_after']
                meets = high_ratio > low_ratio
                status = "[OK]" if meets else "[X]"
                print(f"{subject:<12} {status} Low: {low_ratio:.3f}, High: {high_ratio:.3f}")
                if meets:
                    subjects_meeting += 1
        
        print(f"\nTotal cumpliendo hipotesis: {subjects_meeting}/{len(df_summary['subject'].unique())}")

def main():
    """Función principal."""
    print("="*80)
    print("PASO 4: CALCULO DE COGNITIVE LOAD CON DATOS LIMPIADOS")
    print("="*80)
    print("\nAplicando limpieza de artefactos y recalculando ratios...")
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
            result = analyze_subject_cleaned(csv_file, OUTPUT_DIR)
            all_results.append(result)
        except Exception as e:
            print(f"\nERROR analizando {csv_file.name}: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # Crear visualización comparativa
    if all_results:
        create_comparison_visualization(all_results, OUTPUT_DIR)
        create_final_report(all_results, OUTPUT_DIR)
    
    print("\n" + "="*80)
    print("ANALISIS COMPLETADO")
    print("="*80)
    print(f"\nTotal de sujetos analizados: {len(all_results)}")
    print(f"Resultados guardados en: {OUTPUT_DIR}")

if __name__ == '__main__':
    main()
