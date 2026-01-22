"""
Script de debug para investigar el problema con los datos de Edwin.
"""

import pandas as pd
import numpy as np
from scipy import signal
from pathlib import Path

import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Configuración
CSV_PATH = os.path.join(BASE_DIR, 'data', 'DATA', 'Data-Experimento-Rafa', 'data_Edwin', 'eeg_data_20260120_130753.csv')
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

def calculate_spectral_analysis(signal_data, sample_rate):
    """Análisis espectral detallado."""
    if len(signal_data) < sample_rate // 2:
        return None
    
    nperseg = min(len(signal_data), sample_rate)
    freqs, psd = signal.welch(signal_data, sample_rate, nperseg=nperseg, noverlap=nperseg//2)
    
    # Bandas de frecuencia
    idx_theta = np.logical_and(freqs >= 4.0, freqs <= 7.0)
    idx_alpha = np.logical_and(freqs >= 8.0, freqs <= 12.0)
    idx_beta = np.logical_and(freqs >= 13.0, freqs <= 30.0)
    idx_delta = np.logical_and(freqs >= 1.0, freqs <= 3.0)
    idx_wide = np.logical_and(freqs >= 1.0, freqs <= 40.0)
    
    results = {
        'freqs': freqs,
        'psd': psd,
        'theta_power': np.trapz(psd[idx_theta], freqs[idx_theta]) if np.sum(idx_theta) > 0 else 0,
        'alpha_power': np.trapz(psd[idx_alpha], freqs[idx_alpha]) if np.sum(idx_alpha) > 0 else 0,
        'beta_power': np.trapz(psd[idx_beta], freqs[idx_beta]) if np.sum(idx_beta) > 0 else 0,
        'delta_power': np.trapz(psd[idx_delta], freqs[idx_delta]) if np.sum(idx_delta) > 0 else 0,
        'total_power': np.trapz(psd[idx_wide], freqs[idx_wide]) if np.sum(idx_wide) > 0 else 0
    }
    
    return results

def preprocess_phase(df, label):
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
    print(f"  Fz: mean={np.mean(eeg_matrix[:, FZ_CHANNEL]):.2f}, std={np.std(eeg_matrix[:, FZ_CHANNEL]):.2f}, range=[{np.min(eeg_matrix[:, FZ_CHANNEL]):.2f}, {np.max(eeg_matrix[:, FZ_CHANNEL]):.2f}]")
    print(f"  Pz: mean={np.mean(eeg_matrix[:, PZ_CHANNEL]):.2f}, std={np.std(eeg_matrix[:, PZ_CHANNEL]):.2f}, range=[{np.min(eeg_matrix[:, PZ_CHANNEL]):.2f}, {np.max(eeg_matrix[:, PZ_CHANNEL]):.2f}]")
    
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
    
    # Extraer Fz y Pz
    fz_clean = eeg_matrix[:, FZ_CHANNEL]
    pz_clean = eeg_matrix[:, PZ_CHANNEL]
    
    return fz_clean, pz_clean

def analyze_phase_spectral(fz_signal, pz_signal, label):
    """Análisis espectral detallado de una fase."""
    print(f"\n{'='*70}")
    print(f"ANÁLISIS ESPECTRAL: {label}")
    print(f"{'='*70}")
    
    # Análisis Fz (Theta)
    print(f"\nCanal Fz (Theta 4-7 Hz):")
    fz_spectral = calculate_spectral_analysis(fz_signal, SAMPLE_RATE)
    if fz_spectral:
        print(f"  Theta Power (4-7 Hz): {fz_spectral['theta_power']:.6f}")
        print(f"  Alpha Power (8-12 Hz): {fz_spectral['alpha_power']:.6f}")
        print(f"  Beta Power (13-30 Hz): {fz_spectral['beta_power']:.6f}")
        print(f"  Delta Power (1-3 Hz): {fz_spectral['delta_power']:.6f}")
        print(f"  Total Power (1-40 Hz): {fz_spectral['total_power']:.6f}")
        if fz_spectral['total_power'] > 0:
            print(f"  Theta %: {fz_spectral['theta_power']/fz_spectral['total_power']*100:.2f}%")
            print(f"  Alpha %: {fz_spectral['alpha_power']/fz_spectral['total_power']*100:.2f}%")
    
    # Análisis Pz (Alpha)
    print(f"\nCanal Pz (Alpha 8-12 Hz):")
    pz_spectral = calculate_spectral_analysis(pz_signal, SAMPLE_RATE)
    if pz_spectral:
        print(f"  Theta Power (4-7 Hz): {pz_spectral['theta_power']:.6f}")
        print(f"  Alpha Power (8-12 Hz): {pz_spectral['alpha_power']:.6f}")
        print(f"  Beta Power (13-30 Hz): {pz_spectral['beta_power']:.6f}")
        print(f"  Delta Power (1-3 Hz): {pz_spectral['delta_power']:.6f}")
        print(f"  Total Power (1-40 Hz): {pz_spectral['total_power']:.6f}")
        if pz_spectral['total_power'] > 0:
            print(f"  Theta %: {pz_spectral['theta_power']/pz_spectral['total_power']*100:.2f}%")
            print(f"  Alpha %: {pz_spectral['alpha_power']/pz_spectral['total_power']*100:.2f}%")
    
    # Calcular ratio
    if fz_spectral and pz_spectral:
        theta_power = fz_spectral['theta_power']
        alpha_power = pz_spectral['alpha_power']
        
        if alpha_power > 0:
            ratio = theta_power / alpha_power
            print(f"\nRatio Cognitive Load (Theta_Fz / Alpha_Pz): {ratio:.6f}")
            return ratio, theta_power, alpha_power
    
    return None, None, None

def main():
    """Función principal."""
    print("="*70)
    print("DEBUG: PROBLEMA CON EDWIN")
    print("="*70)
    
    # Cargar datos
    print(f"\nCargando datos de {CSV_PATH}...")
    df = pd.read_csv(CSV_PATH)
    print(f"Total de muestras: {len(df)}")
    print(f"Labels disponibles: {df['label'].unique()}")
    
    # Analizar fases de interés
    phases = ['low_cognitive_load', 'high_cognitive_load']
    
    results = {}
    
    for phase in phases:
        # Preprocesar
        fz, pz = preprocess_phase(df, phase)
        
        if fz is None or pz is None:
            results[phase] = None
            continue
        
        # Análisis espectral
        ratio, theta, alpha = analyze_phase_spectral(fz, pz, phase)
        results[phase] = {
            'ratio': ratio,
            'theta_power': theta,
            'alpha_power': alpha,
            'fz_signal': fz,
            'pz_signal': pz
        }
    
    # Comparación detallada
    print("\n" + "="*70)
    print("COMPARACIÓN DETALLADA")
    print("="*70)
    
    if results['low_cognitive_load'] and results['high_cognitive_load']:
        low = results['low_cognitive_load']
        high = results['high_cognitive_load']
        
        print(f"\nLow Cognitive Load:")
        print(f"  Theta Power (Fz): {low['theta_power']:.6f}")
        print(f"  Alpha Power (Pz): {low['alpha_power']:.6f}")
        print(f"  Ratio: {low['ratio']:.6f}")
        
        print(f"\nHigh Cognitive Load:")
        print(f"  Theta Power (Fz): {high['theta_power']:.6f}")
        print(f"  Alpha Power (Pz): {high['alpha_power']:.6f}")
        print(f"  Ratio: {high['ratio']:.6f}")
        
        print(f"\nCambios:")
        theta_change = high['theta_power'] - low['theta_power']
        alpha_change = high['alpha_power'] - low['alpha_power']
        ratio_change = high['ratio'] - low['ratio']
        
        print(f"  Theta: {low['theta_power']:.6f} -> {high['theta_power']:.6f} (cambio: {theta_change:+.6f}, {theta_change/low['theta_power']*100:+.1f}%)")
        print(f"  Alpha: {low['alpha_power']:.6f} -> {high['alpha_power']:.6f} (cambio: {alpha_change:+.6f}, {alpha_change/low['alpha_power']*100:+.1f}%)")
        print(f"  Ratio: {low['ratio']:.6f} -> {high['ratio']:.6f} (cambio: {ratio_change:+.6f})")
        
        print(f"\nInterpretación:")
        if theta_change < 0:
            print(f"  [HALLAZGO] Theta DISMINUYE en High Load ({theta_change:.6f})")
        else:
            print(f"  [OK] Theta AUMENTA en High Load ({theta_change:.6f})")
        
        if alpha_change > 0:
            print(f"  [HALLAZGO] Alpha AUMENTA en High Load ({alpha_change:.6f})")
            print(f"  [HALLAZGO] Esto causa que el ratio disminuya (Alpha aumenta más que Theta)")
        elif alpha_change < 0:
            print(f"  [OK] Alpha DISMINUYE en High Load ({alpha_change:.6f})")
        
        if ratio_change < 0:
            print(f"  [PROBLEMA] El ratio DISMINUYE de Low a High ({ratio_change:.6f})")
            print(f"  [PROBLEMA] Esto contradice la hipótesis esperada")
            print(f"  [POSIBLE CAUSA] Alpha aumenta demasiado o Theta disminuye demasiado en High Load")
    
    print("\n" + "="*70)

if __name__ == '__main__':
    main()
