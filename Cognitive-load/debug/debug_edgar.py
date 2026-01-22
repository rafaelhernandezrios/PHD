"""
Script de debug para investigar el problema con los datos de Edgar.
"""

import pandas as pd
import numpy as np
from scipy import signal
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from core.waaf_filter import waaf_filter_2d, HAS_WAVELETS

import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Configuración
CSV_PATH = os.path.join(BASE_DIR, 'data', 'DATA', 'Data-Experimento-Rafa', 'data_Edgar', 'eeg_data_20260120_140754.csv')
SAMPLE_RATE = 250
THETA_BAND = (4.0, 7.0)
ALPHA_BAND = (8.0, 12.0)
FZ_CHANNEL = 3
PZ_CHANNEL = 6

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
    """Aplica Common Average Reference."""
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
    """Calcula la potencia espectral."""
    if len(signal_data) < sample_rate // 2:
        return 0.0
    
    nperseg = min(len(signal_data), sample_rate)
    freqs, psd = signal.welch(signal_data, sample_rate, nperseg=nperseg, noverlap=nperseg//2)
    
    idx_band = np.logical_and(freqs >= freq_band[0], freqs <= freq_band[1])
    
    if np.sum(idx_band) == 0:
        return 0.0
    
    bandpower = np.trapz(psd[idx_band], freqs[idx_band])
    return bandpower

def preprocess_phase(df, label, apply_waaf=False):
    """Preprocesa una fase específica."""
    label_data = df[df['label'] == label].copy()
    
    print(f"\n{'='*70}")
    print(f"PROCESANDO: {label}")
    print(f"{'='*70}")
    print(f"Muestras disponibles: {len(label_data)}")
    
    if len(label_data) < 125:
        print(f"ERROR: Muy pocas muestras ({len(label_data)} < 125)")
        return None, None
    
    n_samples = len(label_data)
    n_channels = 8
    
    # Construir matriz EEG
    eeg_matrix = np.zeros((n_samples, n_channels))
    for ch in range(n_channels):
        eeg_matrix[:, ch] = label_data[f'channel_{ch}'].values
    
    print(f"\nEstadísticas ANTES de filtros:")
    print(f"  Fz: mean={np.mean(eeg_matrix[:, FZ_CHANNEL]):.2f}, std={np.std(eeg_matrix[:, FZ_CHANNEL]):.2f}")
    print(f"  Pz: mean={np.mean(eeg_matrix[:, PZ_CHANNEL]):.2f}, std={np.std(eeg_matrix[:, PZ_CHANNEL]):.2f}")
    
    # Paso 1: Filtros temporales
    print(f"\nAplicando filtros temporales...")
    for ch in range(n_channels):
        eeg_matrix[:, ch] = apply_notch_filter(eeg_matrix[:, ch], 60.0, SAMPLE_RATE)
        eeg_matrix[:, ch] = apply_bandpass_filter(eeg_matrix[:, ch], 1.0, 40.0, SAMPLE_RATE)
    
    print(f"Estadísticas DESPUÉS de filtros:")
    print(f"  Fz: mean={np.mean(eeg_matrix[:, FZ_CHANNEL]):.2f}, std={np.std(eeg_matrix[:, FZ_CHANNEL]):.2f}")
    print(f"  Pz: mean={np.mean(eeg_matrix[:, PZ_CHANNEL]):.2f}, std={np.std(eeg_matrix[:, PZ_CHANNEL]):.2f}")
    
    # Paso 2: CAR
    print(f"\nAplicando CAR...")
    eeg_matrix = apply_car(eeg_matrix, bad_channel_idx=None)
    
    print(f"Estadísticas DESPUÉS de CAR:")
    print(f"  Fz: mean={np.mean(eeg_matrix[:, FZ_CHANNEL]):.2f}, std={np.std(eeg_matrix[:, FZ_CHANNEL]):.2f}")
    print(f"  Pz: mean={np.mean(eeg_matrix[:, PZ_CHANNEL]):.2f}, std={np.std(eeg_matrix[:, PZ_CHANNEL]):.2f}")
    
    # Paso 3: WAAF (opcional)
    if apply_waaf and HAS_WAVELETS:
        print(f"\nAplicando WAAF...")
        try:
            eeg_matrix, _, _ = waaf_filter_2d(eeg_matrix, wavelet='db4', level=7, attn=[1,2,3,4])
            print(f"WAAF aplicado exitosamente")
            print(f"Estadísticas DESPUÉS de WAAF:")
            print(f"  Fz: mean={np.mean(eeg_matrix[:, FZ_CHANNEL]):.2f}, std={np.std(eeg_matrix[:, FZ_CHANNEL]):.2f}")
            print(f"  Pz: mean={np.mean(eeg_matrix[:, PZ_CHANNEL]):.2f}, std={np.std(eeg_matrix[:, PZ_CHANNEL]):.2f}")
        except Exception as e:
            print(f"ERROR aplicando WAAF: {e}")
            import traceback
            traceback.print_exc()
    
    # Extraer Fz y Pz
    fz_clean = eeg_matrix[:, FZ_CHANNEL]
    pz_clean = eeg_matrix[:, PZ_CHANNEL]
    
    # Verificar valores
    print(f"\nVerificación final:")
    print(f"  Fz: len={len(fz_clean)}, min={np.min(fz_clean):.2f}, max={np.max(fz_clean):.2f}")
    print(f"  Pz: len={len(pz_clean)}, min={np.min(pz_clean):.2f}, max={np.max(pz_clean):.2f}")
    print(f"  Fz tiene NaN: {np.any(np.isnan(fz_clean))}")
    print(f"  Pz tiene NaN: {np.any(np.isnan(pz_clean))}")
    print(f"  Fz tiene Inf: {np.any(np.isinf(fz_clean))}")
    print(f"  Pz tiene Inf: {np.any(np.isinf(pz_clean))}")
    
    return fz_clean, pz_clean

def calculate_ratio_for_phase(fz_signal, pz_signal):
    """Calcula el ratio de cognitive load para una señal completa."""
    if fz_signal is None or pz_signal is None:
        return None
    
    if len(fz_signal) < 125:
        print(f"ERROR: Señal muy corta ({len(fz_signal)} muestras)")
        return None
    
    print(f"\nCalculando potencia espectral...")
    
    # Calcular potencias
    theta_power = calculate_bandpower(fz_signal, THETA_BAND, SAMPLE_RATE)
    alpha_power = calculate_bandpower(pz_signal, ALPHA_BAND, SAMPLE_RATE)
    
    print(f"  Theta Power (Fz): {theta_power:.6f}")
    print(f"  Alpha Power (Pz): {alpha_power:.6f}")
    
    # Calcular ratio
    if alpha_power > 0 and np.isfinite(theta_power) and np.isfinite(alpha_power):
        ratio = theta_power / alpha_power
        print(f"  Ratio (Theta/Alpha): {ratio:.6f}")
        
        # Verificar filtro de cordura (ajustado para permitir ratios más bajos)
        if ratio < 0.01:
            print(f"  ADVERTENCIA: Ratio extremadamente bajo ({ratio:.6f} < 0.01) - posible error")
            return None
        elif ratio > 100:
            print(f"  ADVERTENCIA: Ratio extremadamente alto ({ratio:.6f} > 100) - posible error")
            return None
        else:
            if ratio < 0.1:
                print(f"  Ratio bajo pero válido: {ratio:.6f} (< 0.1, pero aceptado)")
            else:
                print(f"  Ratio válido: {ratio:.6f}")
            return ratio
    else:
        print(f"  ERROR: No se pudo calcular ratio")
        print(f"    alpha_power > 0: {alpha_power > 0}")
        print(f"    theta_power finito: {np.isfinite(theta_power)}")
        print(f"    alpha_power finito: {np.isfinite(alpha_power)}")
        return None

def main():
    """Función principal."""
    print("="*70)
    print("DEBUG: PROBLEMA CON EDGAR")
    print("="*70)
    
    # Cargar datos
    print(f"\nCargando datos de {CSV_PATH}...")
    df = pd.read_csv(CSV_PATH)
    print(f"Total de muestras: {len(df)}")
    print(f"Labels disponibles: {df['label'].unique()}")
    
    # Analizar fases de interés
    phases = ['low_cognitive_load', 'high_cognitive_load']
    
    print("\n" + "="*70)
    print("ANALISIS SIN WAAF")
    print("="*70)
    
    results_without = {}
    for phase in phases:
        fz, pz = preprocess_phase(df, phase, apply_waaf=False)
        ratio = calculate_ratio_for_phase(fz, pz)
        results_without[phase] = ratio
    
    print("\n" + "="*70)
    print("ANALISIS CON WAAF")
    print("="*70)
    
    results_with = {}
    for phase in phases:
        fz, pz = preprocess_phase(df, phase, apply_waaf=True)
        ratio = calculate_ratio_for_phase(fz, pz)
        results_with[phase] = ratio
    
    # Resumen
    print("\n" + "="*70)
    print("RESUMEN")
    print("="*70)
    
    print("\nSin WAAF:")
    for phase, ratio in results_without.items():
        if ratio is not None:
            print(f"  {phase}: {ratio:.6f}")
        else:
            print(f"  {phase}: ERROR - No se pudo calcular")
    
    print("\nCon WAAF:")
    for phase, ratio in results_with.items():
        if ratio is not None:
            print(f"  {phase}: {ratio:.6f}")
        else:
            print(f"  {phase}: ERROR - No se pudo calcular")
    
    # Verificar hipótesis
    if (results_without['high_cognitive_load'] is not None and 
        results_without['low_cognitive_load'] is not None):
        high = results_without['high_cognitive_load']
        low = results_without['low_cognitive_load']
        print(f"\nHipotesis (sin WAAF): High ({high:.6f}) > Low ({low:.6f})? {high > low}")
    
    if (results_with['high_cognitive_load'] is not None and 
        results_with['low_cognitive_load'] is not None):
        high = results_with['high_cognitive_load']
        low = results_with['low_cognitive_load']
        print(f"Hipotesis (con WAAF): High ({high:.6f}) > Low ({low:.6f})? {high > low}")

if __name__ == '__main__':
    main()
