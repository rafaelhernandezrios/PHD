"""
Script para analizar cognitive load de todos los sujetos y comparar resultados.
Identifica diferencias entre sujetos donde se cumple la hipótesis vs donde no.
"""

import pandas as pd
import numpy as np
from scipy import signal
from pathlib import Path
import glob
import os

# Configuración
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, 'output', 'analysis_output')
DATA_DIR = os.path.join(BASE_DIR, 'data')
SAMPLE_RATE = 250  # Hz
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
    """Calcula la potencia espectral en una banda de frecuencias usando el método de Welch."""
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

def preprocess_phase_data(df, label):
    """
    Preprocesa datos de una fase: Filtros + CAR (sin WAAF).
    
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
    
    # Paso 1: Filtros temporales
    for ch in range(n_channels):
        eeg_matrix[:, ch] = apply_notch_filter(eeg_matrix[:, ch], 60.0, SAMPLE_RATE)
        eeg_matrix[:, ch] = apply_bandpass_filter(eeg_matrix[:, ch], 1.0, 40.0, SAMPLE_RATE)
    
    # Paso 2: CAR
    eeg_matrix = apply_car(eeg_matrix, bad_channel_idx=None)
    
    # Extraer Fz y Pz
    fz_clean = eeg_matrix[:, FZ_CHANNEL]
    pz_clean = eeg_matrix[:, PZ_CHANNEL]
    
    return fz_clean, pz_clean

def calculate_cognitive_load_by_phase(df, labels_of_interest):
    """Calcula la carga cognitiva (ratio Theta/Alpha) para cada fase con preprocesamiento."""
    results = {}
    min_samples = SAMPLE_RATE // 2
    
    for label in labels_of_interest:
        label_data = df[df['label'] == label].copy()
        
        if len(label_data) < min_samples:
            results[label] = []
            continue
        
        # Preprocesar datos (filtros + CAR, sin WAAF)
        fz_clean, pz_clean = preprocess_phase_data(df, label)
        
        if fz_clean is None or pz_clean is None:
            results[label] = []
            continue
        
        ratios = []
        
        # Ajustar tamaño de ventana según datos disponibles
        if len(fz_clean) >= WINDOW_SAMPLES:
            actual_window = WINDOW_SAMPLES
            step_samples = WINDOW_SAMPLES // 2
        else:
            actual_window = len(fz_clean)
            step_samples = actual_window
        
        # Ventaneo deslizante
        for start_idx in range(0, len(fz_clean) - actual_window + 1, step_samples):
            end_idx = start_idx + actual_window
            
            fz_window = fz_clean[start_idx:end_idx]
            pz_window = pz_clean[start_idx:end_idx]
            
            theta_power = calculate_bandpower(fz_window, THETA_BAND, SAMPLE_RATE)
            alpha_power = calculate_bandpower(pz_window, ALPHA_BAND, SAMPLE_RATE)
            
            if alpha_power > 0 and np.isfinite(theta_power) and np.isfinite(alpha_power):
                ratio = theta_power / alpha_power
                # Ajustar filtro de cordura para permitir ratios más bajos
                if np.isfinite(ratio) and 0.01 < ratio < 100:
                    ratios.append(ratio)
        
        results[label] = ratios
    
    return results

def analyze_subject(csv_path):
    """Analiza un sujeto y retorna estadísticas."""
    try:
        df = pd.read_csv(csv_path)
        subject_name = Path(csv_path).parent.name.replace('data_', '')
        
        # Calcular ratios
        ratios = calculate_cognitive_load_by_phase(df, LABELS_OF_INTEREST)
        
        # Extraer estadísticas
        stats = {}
        for label in LABELS_OF_INTEREST:
            if label in ratios and len(ratios[label]) > 0:
                stats[label] = {
                    'mean': np.mean(ratios[label]),
                    'median': np.median(ratios[label]),
                    'std': np.std(ratios[label]),
                    'n': len(ratios[label]),
                    'min': np.min(ratios[label]),
                    'max': np.max(ratios[label])
                }
            else:
                stats[label] = None
        
        # Verificar hipótesis: High Load > Low Load
        hypothesis_met = False
        if 'high_cognitive_load' in stats and 'low_cognitive_load' in stats:
            if stats['high_cognitive_load'] and stats['low_cognitive_load']:
                high_mean = stats['high_cognitive_load']['mean']
                low_mean = stats['low_cognitive_load']['mean']
                hypothesis_met = high_mean > low_mean
        
        return {
            'subject': subject_name,
            'csv_path': csv_path,
            'stats': stats,
            'hypothesis_met': hypothesis_met,
            'total_samples': len(df)
        }
    except Exception as e:
        print(f"Error analizando {csv_path}: {str(e)}")
        return None

def main():
    """Función principal."""
    print("="*70)
    print("ANALISIS DE CARGA COGNITIVA - TODOS LOS SUJETOS")
    print("="*70)
    
    # Encontrar todos los archivos CSV
    csv_files = []
    # Buscar en DATA/DATA-Experimento-Rafa y subdirectorios
    data_path = Path(DATA_DIR)
    if (data_path / 'DATA' / 'Data-Experimento-Rafa').exists():
        for data_dir in (data_path / 'DATA' / 'Data-Experimento-Rafa').glob('data_*'):
            csv_files.extend(data_dir.glob('eeg_data_*.csv'))
    # También buscar en el directorio raíz de data si hay carpetas data_*
    for data_dir in data_path.glob('data_*'):
        csv_files.extend(data_dir.glob('eeg_data_*.csv'))
    
    if not csv_files:
        print("No se encontraron archivos CSV en carpetas data_*")
        return
    
    print(f"\nEncontrados {len(csv_files)} archivos CSV:")
    for csv_file in csv_files:
        print(f"  - {csv_file}")
    
    # Analizar cada sujeto
    print("\n" + "="*70)
    print("ANALIZANDO SUJETOS...")
    print("="*70)
    
    all_results = []
    for csv_file in sorted(csv_files):
        result = analyze_subject(csv_file)
        if result:
            all_results.append(result)
    
    # Mostrar resultados
    print("\n" + "="*70)
    print("RESULTADOS POR SUJETO")
    print("="*70)
    
    for result in all_results:
        print(f"\n{result['subject'].upper()}:")
        print(f"  Total muestras: {result['total_samples']}")
        print(f"  Hipotesis cumplida (High > Low): {result['hypothesis_met']}")
        print(f"\n  Ratios por fase:")
        
        for label in LABELS_OF_INTEREST:
            if result['stats'][label]:
                s = result['stats'][label]
                print(f"    {label}:")
                print(f"      Mean: {s['mean']:.3f}")
                print(f"      Median: {s['median']:.3f}")
                print(f"      Std: {s['std']:.3f}")
                print(f"      N: {s['n']}")
                print(f"      Range: [{s['min']:.3f}, {s['max']:.3f}]")
            else:
                print(f"    {label}: Sin datos")
    
    # Comparación
    print("\n" + "="*70)
    print("COMPARACION Y CONCLUSIONES")
    print("="*70)
    
    subjects_met = [r for r in all_results if r['hypothesis_met']]
    subjects_not_met = [r for r in all_results if not r['hypothesis_met']]
    
    print(f"\nSujetos donde se CUMPLE la hipotesis (High > Low): {len(subjects_met)}")
    for r in subjects_met:
        high = r['stats']['high_cognitive_load']['mean'] if r['stats']['high_cognitive_load'] else 0
        low = r['stats']['low_cognitive_load']['mean'] if r['stats']['low_cognitive_load'] else 0
        print(f"  - {r['subject']}: High={high:.3f} > Low={low:.3f} (diff: {high-low:.3f})")
    
    print(f"\nSujetos donde NO se cumple la hipotesis (High <= Low): {len(subjects_not_met)}")
    for r in subjects_not_met:
        high = r['stats']['high_cognitive_load']['mean'] if r['stats']['high_cognitive_load'] else 0
        low = r['stats']['low_cognitive_load']['mean'] if r['stats']['low_cognitive_load'] else 0
        print(f"  - {r['subject']}: High={high:.3f} <= Low={low:.3f} (diff: {high-low:.3f})")
    
    # Análisis de diferencias
    print("\n" + "="*70)
    print("ANALISIS DE DIFERENCIAS")
    print("="*70)
    
    if subjects_met and subjects_not_met:
        print("\nComparando sujetos que cumplen vs no cumplen:")
        
        # Promedio de ratios High Load
        high_met = np.mean([r['stats']['high_cognitive_load']['mean'] 
                           for r in subjects_met if r['stats']['high_cognitive_load']])
        high_not_met = np.mean([r['stats']['high_cognitive_load']['mean'] 
                               for r in subjects_not_met if r['stats']['high_cognitive_load']])
        
        # Promedio de ratios Low Load
        low_met = np.mean([r['stats']['low_cognitive_load']['mean'] 
                          for r in subjects_met if r['stats']['low_cognitive_load']])
        low_not_met = np.mean([r['stats']['low_cognitive_load']['mean'] 
                              for r in subjects_not_met if r['stats']['low_cognitive_load']])
        
        print(f"\n  Promedio High Load:")
        print(f"    Sujetos que cumplen: {high_met:.3f}")
        print(f"    Sujetos que NO cumplen: {high_not_met:.3f}")
        print(f"    Diferencia: {high_met - high_not_met:.3f}")
        
        print(f"\n  Promedio Low Load:")
        print(f"    Sujetos que cumplen: {low_met:.3f}")
        print(f"    Sujetos que NO cumplen: {low_not_met:.3f}")
        print(f"    Diferencia: {low_met - low_not_met:.3f}")
        
        # Verificar número de ventanas
        print(f"\n  Numero de ventanas (N):")
        for r in all_results:
            if r['stats']['high_cognitive_load']:
                n_high = r['stats']['high_cognitive_load']['n']
                n_low = r['stats']['low_cognitive_load']['n'] if r['stats']['low_cognitive_load'] else 0
                print(f"    {r['subject']}: High={n_high}, Low={n_low}")
    
    # Tabla resumen
    print("\n" + "="*70)
    print("TABLA RESUMEN")
    print("="*70)
    print(f"\n{'Sujeto':<15} {'High Load':<12} {'Low Load':<12} {'Diff':<12} {'Hipotesis':<12}")
    print("-" * 70)
    for r in all_results:
        high = r['stats']['high_cognitive_load']['mean'] if r['stats']['high_cognitive_load'] else 0
        low = r['stats']['low_cognitive_load']['mean'] if r['stats']['low_cognitive_load'] else 0
        diff = high - low
        status = "CUMPLE" if r['hypothesis_met'] else "NO CUMPLE"
        print(f"{r['subject']:<15} {high:<12.3f} {low:<12.3f} {diff:<12.3f} {status:<12}")
    
    print("\n" + "="*70)
    print("ANALISIS COMPLETADO")
    print("="*70)

if __name__ == '__main__':
    main()
