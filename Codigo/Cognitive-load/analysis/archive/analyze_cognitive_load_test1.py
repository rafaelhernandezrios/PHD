"""
Análisis completo de cognitive load para data_test1
Calcula ratio Theta/Alpha por fase y genera visualizaciones
"""

import pandas as pd
import numpy as np
from scipy import signal
from pathlib import Path
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Configuración
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(BASE_DIR, 'data_test1', 'eeg_data_20260121_225034.csv')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output', 'analysis_output')
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# Parámetros
SAMPLE_RATE = 250  # Hz (frecuencia original)
EFFECTIVE_SAMPLE_RATE = 50  # Hz (con subsampling, pero parece que no se aplicó)
THETA_BAND = (4.0, 7.0)  # Hz
ALPHA_BAND = (8.0, 12.0)  # Hz
WINDOW_DURATION = 2.0  # segundos
WINDOW_SAMPLES = int(WINDOW_DURATION * SAMPLE_RATE)  # 500 muestras
FZ_CHANNEL = 3  # Canal Fz (frontal)
PZ_CHANNEL = 6  # Canal Pz (parietal)

# Labels de interés
LABELS_OF_INTEREST = [
    'baseline_eyes_open',
    'baseline_eyes_closed',
    'low_cognitive_load',
    'high_cognitive_load'
]

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

def apply_bandpass_filter(signal_data, low_freq=1.0, high_freq=40.0, sample_rate=250):
    """Aplica filtro bandpass."""
    nyquist = sample_rate / 2
    low = low_freq / nyquist
    high = high_freq / nyquist
    b, a = signal.butter(4, [low, high], btype='band')
    filtered = signal.filtfilt(b, a, signal_data)
    return filtered

def apply_notch_filter(signal_data, notch_freq=60.0, sample_rate=250, Q=30.0):
    """Aplica filtro notch."""
    b, a = signal.iirnotch(notch_freq, Q, sample_rate)
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

def preprocess_phase_data(df, label, sample_rate=250):
    """
    Preprocesa datos de una fase: Filtros + CAR.
    
    Returns:
        fz_clean, pz_clean: Señales preprocesadas
    """
    label_data = df[df['label'] == label].copy()
    
    if len(label_data) < 125:  # Mínimo 0.5 segundos
        return None, None
    
    n_samples = len(label_data)
    n_channels = 8
    
    # Construir matriz EEG [n_samples, n_channels]
    eeg_matrix = np.zeros((n_samples, n_channels))
    for ch in range(n_channels):
        eeg_matrix[:, ch] = label_data[f'channel_{ch}'].values
    
    # Aplicar filtros
    eeg_filtered = np.zeros_like(eeg_matrix)
    for ch in range(n_channels):
        # Notch 60 Hz
        notch_filtered = apply_notch_filter(eeg_matrix[:, ch], sample_rate=sample_rate)
        # Bandpass 1-40 Hz
        eeg_filtered[:, ch] = apply_bandpass_filter(notch_filtered, sample_rate=sample_rate)
    
    # Aplicar CAR
    eeg_car = apply_car(eeg_filtered, bad_channel_idx=None)
    
    # Extraer Fz y Pz
    fz_clean = eeg_car[:, FZ_CHANNEL]
    pz_clean = eeg_car[:, PZ_CHANNEL]
    
    return fz_clean, pz_clean

def calculate_cognitive_load_by_phase(df, labels_of_interest, sample_rate=250):
    """
    Calcula la carga cognitiva (ratio Theta/Alpha) para cada fase usando ventanas deslizantes.
    """
    results = {}
    min_samples = sample_rate // 2  # Mínimo 0.5 segundos
    
    for label in labels_of_interest:
        print(f"\nProcesando {label}...")
        
        # Preprocesar datos de la fase
        fz_clean, pz_clean = preprocess_phase_data(df, label, sample_rate)
        
        if fz_clean is None or pz_clean is None:
            print(f"  Advertencia: {label} tiene muy pocos datos")
            results[label] = {'ratios': [], 'theta_powers': [], 'alpha_powers': []}
            continue
        
        print(f"  Datos preprocesados: {len(fz_clean)} muestras ({len(fz_clean)/sample_rate:.1f} seg)")
        
        # Calcular ratio usando ventanas deslizantes
        ratios = []
        theta_powers = []
        alpha_powers = []
        
        # Ajustar tamaño de ventana según datos disponibles
        if len(fz_clean) >= WINDOW_SAMPLES:
            actual_window = WINDOW_SAMPLES
            step_samples = WINDOW_SAMPLES // 2  # 50% overlap
        else:
            actual_window = len(fz_clean)
            step_samples = actual_window
        
        print(f"  Ventana: {actual_window} muestras ({actual_window/sample_rate:.1f} seg), paso: {step_samples} muestras")
        
        n_windows = 0
        for start_idx in range(0, len(fz_clean) - actual_window + 1, step_samples):
            end_idx = start_idx + actual_window
            fz_window = fz_clean[start_idx:end_idx]
            pz_window = pz_clean[start_idx:end_idx]
            
            theta_power = calculate_bandpower(fz_window, THETA_BAND, sample_rate)
            alpha_power = calculate_bandpower(pz_window, ALPHA_BAND, sample_rate)
            
            if alpha_power > 0 and np.isfinite(theta_power) and np.isfinite(alpha_power):
                ratio = theta_power / alpha_power
                if np.isfinite(ratio) and 0.01 < ratio < 100:  # Sanity filter
                    ratios.append(ratio)
                    theta_powers.append(theta_power)
                    alpha_powers.append(alpha_power)
                    n_windows += 1
        
        print(f"  Ventanas válidas: {n_windows}")
        
        if ratios:
            results[label] = {
                'ratios': ratios,
                'theta_powers': theta_powers,
                'alpha_powers': alpha_powers,
                'mean_ratio': np.mean(ratios),
                'median_ratio': np.median(ratios),
                'std_ratio': np.std(ratios),
                'n_windows': n_windows
            }
        else:
            results[label] = {'ratios': [], 'theta_powers': [], 'alpha_powers': []}
    
    return results

def create_visualizations(results, output_dir):
    """Crea visualizaciones de los resultados."""
    print("\nGenerando visualizaciones...")
    
    # 1. Boxplot comparativo
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Cognitive Load Analysis - data_test1', fontsize=16, fontweight='bold')
    
    # Boxplot de ratios
    ax1 = axes[0, 0]
    data_for_box = []
    labels_for_box = []
    for label in LABELS_OF_INTEREST:
        if label in results and results[label]['ratios']:
            data_for_box.append(results[label]['ratios'])
            labels_for_box.append(label.replace('_', '\n'))
    
    if data_for_box:
        bp = ax1.boxplot(data_for_box, labels=labels_for_box, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('#21262d')
            patch.set_edgecolor('#00ff88')
        ax1.set_ylabel('Cognitive Load Ratio (Theta/Alpha)', fontsize=12)
        ax1.set_title('Cognitive Load by Phase', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.set_facecolor('#0d1117')
        ax1.tick_params(colors='white')
        ax1.spines['bottom'].set_color('white')
        ax1.spines['top'].set_color('white')
        ax1.spines['left'].set_color('white')
        ax1.spines['right'].set_color('white')
        ax1.xaxis.label.set_color('white')
        ax1.yaxis.label.set_color('white')
        ax1.title.set_color('white')
    
    # Estadísticas
    ax2 = axes[0, 1]
    ax2.axis('off')
    stats_text = "ESTADÍSTICAS\n" + "="*50 + "\n\n"
    for label in LABELS_OF_INTEREST:
        if label in results and results[label]['ratios']:
            r = results[label]
            stats_text += f"{label.replace('_', ' ').title()}:\n"
            stats_text += f"  Mean: {r['mean_ratio']:.3f}\n"
            stats_text += f"  Median: {r['median_ratio']:.3f}\n"
            stats_text += f"  Std: {r['std_ratio']:.3f}\n"
            stats_text += f"  N: {r['n_windows']}\n"
            stats_text += f"  Range: [{np.min(r['ratios']):.3f}, {np.max(r['ratios']):.3f}]\n\n"
    
    ax2.text(0.1, 0.9, stats_text, transform=ax2.transAxes, 
             fontsize=10, verticalalignment='top', family='monospace',
             bbox=dict(boxstyle='round', facecolor='#21262d', edgecolor='#00ff88', alpha=0.8),
             color='white')
    
    # Distribución de ratios
    ax3 = axes[1, 0]
    for label in LABELS_OF_INTEREST:
        if label in results and results[label]['ratios']:
            ax3.hist(results[label]['ratios'], bins=30, alpha=0.6, 
                    label=label.replace('_', ' ').title(), edgecolor='black')
    ax3.set_xlabel('Cognitive Load Ratio', fontsize=12)
    ax3.set_ylabel('Frequency', fontsize=12)
    ax3.set_title('Distribution of Cognitive Load Ratios', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_facecolor('#0d1117')
    ax3.tick_params(colors='white')
    for spine in ax3.spines.values():
        spine.set_color('white')
    ax3.xaxis.label.set_color('white')
    ax3.yaxis.label.set_color('white')
    ax3.title.set_color('white')
    ax3.legend(facecolor='#21262d', edgecolor='#00ff88', labelcolor='white')
    
    # Comparación High vs Low
    ax4 = axes[1, 1]
    if 'high_cognitive_load' in results and 'low_cognitive_load' in results:
        high_ratios = results['high_cognitive_load'].get('ratios', [])
        low_ratios = results['low_cognitive_load'].get('ratios', [])
        
        if high_ratios and low_ratios:
            comparison_data = [low_ratios, high_ratios]
            bp = ax4.boxplot(comparison_data, labels=['Low Load', 'High Load'], patch_artist=True)
            for patch in bp['boxes']:
                patch.set_facecolor('#21262d')
                patch.set_edgecolor('#00ff88')
            
            # Test estadístico
            from scipy import stats
            if len(high_ratios) > 1 and len(low_ratios) > 1:
                stat, p_value = stats.mannwhitneyu(high_ratios, low_ratios, alternative='greater')
                ax4.text(0.5, 0.95, f'Mann-Whitney U test\np-value: {p_value:.4f}', 
                        transform=ax4.transAxes, ha='center', va='top',
                        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7),
                        fontsize=10, fontweight='bold')
            
            ax4.set_ylabel('Cognitive Load Ratio', fontsize=12)
            ax4.set_title('High vs Low Cognitive Load', fontsize=14, fontweight='bold')
            ax4.grid(True, alpha=0.3)
            ax4.set_facecolor('#0d1117')
            ax4.tick_params(colors='white')
            for spine in ax4.spines.values():
                spine.set_color('white')
            ax4.xaxis.label.set_color('white')
            ax4.yaxis.label.set_color('white')
            ax4.title.set_color('white')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'cognitive_load_test1.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='#0d1117')
    print(f"  Gráfico guardado: {output_path}")
    plt.close()

def main():
    """Función principal."""
    print("="*80)
    print("ANÁLISIS DE CARGA COGNITIVA - data_test1")
    print("="*80)
    
    # Cargar datos
    print("\nCargando datos...")
    df = pd.read_csv(CSV_PATH)
    print(f"Total de muestras: {len(df):,}")
    
    # Determinar frecuencia de muestreo real
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
    
    # Calcular cognitive load por fase
    print("\n" + "="*80)
    print("CALCULANDO CARGA COGNITIVA")
    print("="*80)
    
    results = calculate_cognitive_load_by_phase(df, LABELS_OF_INTEREST, sample_rate=actual_sample_rate)
    
    # Mostrar resultados
    print("\n" + "="*80)
    print("RESULTADOS")
    print("="*80)
    
    for label in LABELS_OF_INTEREST:
        if label in results and results[label].get('ratios'):
            r = results[label]
            print(f"\n{label.replace('_', ' ').title()}:")
            print(f"  Mean ratio: {r['mean_ratio']:.3f}")
            print(f"  Median ratio: {r['median_ratio']:.3f}")
            print(f"  Std: {r['std_ratio']:.3f}")
            print(f"  N ventanas: {r['n_windows']}")
            print(f"  Range: [{np.min(r['ratios']):.3f}, {np.max(r['ratios']):.3f}]")
        else:
            print(f"\n{label.replace('_', ' ').title()}: Sin datos suficientes")
    
    # Comparación High vs Low
    print("\n" + "="*80)
    print("COMPARACIÓN HIGH vs LOW COGNITIVE LOAD")
    print("="*80)
    
    if 'high_cognitive_load' in results and 'low_cognitive_load' in results:
        high_ratios = results['high_cognitive_load'].get('ratios', [])
        low_ratios = results['low_cognitive_load'].get('ratios', [])
        
        if high_ratios and low_ratios:
            high_mean = np.mean(high_ratios)
            low_mean = np.mean(low_ratios)
            diff = high_mean - low_mean
            percentage_diff = (diff / low_mean * 100) if low_mean > 0 else 0
            
            print(f"\nLow Load:  Mean = {low_mean:.3f}")
            print(f"High Load: Mean = {high_mean:.3f}")
            print(f"Diferencia: {diff:.3f} ({percentage_diff:+.1f}%)")
            
            if high_mean > low_mean:
                print("\n[OK] HIPOTESIS CUMPLIDA: High Load > Low Load")
            else:
                print("\n[NO] HIPOTESIS NO CUMPLIDA: High Load <= Low Load")
            
            # Test estadístico
            from scipy import stats
            if len(high_ratios) > 1 and len(low_ratios) > 1:
                stat, p_value = stats.mannwhitneyu(high_ratios, low_ratios, alternative='greater')
                print(f"\nMann-Whitney U test:")
                print(f"  U-statistic: {stat:.2f}")
                print(f"  p-value: {p_value:.6f}")
                if p_value < 0.05:
                    print(f"  -> Diferencia estadisticamente significativa (p < 0.05)")
                else:
                    print(f"  -> Diferencia NO estadisticamente significativa (p >= 0.05)")
    
    # Generar visualizaciones
    create_visualizations(results, OUTPUT_DIR)
    
    # Guardar resultados
    print("\n" + "="*80)
    print("GUARDANDO RESULTADOS")
    print("="*80)
    
    # Crear DataFrame con resultados
    results_data = []
    for label in LABELS_OF_INTEREST:
        if label in results and results[label].get('ratios'):
            for ratio in results[label]['ratios']:
                results_data.append({
                    'phase': label,
                    'cognitive_load_ratio': ratio
                })
    
    if results_data:
        df_results = pd.DataFrame(results_data)
        results_path = os.path.join(OUTPUT_DIR, 'cognitive_load_test1_results.csv')
        df_results.to_csv(results_path, index=False)
        print(f"Resultados guardados: {results_path}")
    
    print("\n" + "="*80)
    print("ANÁLISIS COMPLETADO")
    print("="*80)

if __name__ == '__main__':
    main()
