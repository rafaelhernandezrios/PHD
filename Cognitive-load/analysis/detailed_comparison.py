"""
Análisis detallado comparando sujetos que cumplen vs no cumplen la hipótesis.
"""

import pandas as pd
import numpy as np
from scipy import signal
from pathlib import Path

def analyze_signal_quality(csv_path):
    """Analiza la calidad de la señal y estadísticas básicas."""
    df = pd.read_csv(csv_path)
    subject = Path(csv_path).parent.name.replace('data_', '')
    
    results = {}
    
    for label in ['low_cognitive_load', 'high_cognitive_load']:
        label_data = df[df['label'] == label]
        
        if len(label_data) == 0:
            continue
        
        # Extraer Fz y Pz
        fz = label_data['channel_3'].values
        pz = label_data['channel_6'].values
        
        # Estadísticas básicas
        results[label] = {
            'n_samples': len(label_data),
            'fz_mean': np.mean(fz),
            'fz_std': np.std(fz),
            'fz_min': np.min(fz),
            'fz_max': np.max(fz),
            'pz_mean': np.mean(pz),
            'pz_std': np.std(pz),
            'pz_min': np.min(pz),
            'pz_max': np.max(pz),
            'fz_range': np.max(fz) - np.min(fz),
            'pz_range': np.max(pz) - np.min(pz)
        }
    
    return subject, results

def calculate_spectral_features(csv_path):
    """Calcula características espectrales de las señales."""
    df = pd.read_csv(csv_path)
    subject = Path(csv_path).parent.name.replace('data_', '')
    
    results = {}
    
    for label in ['low_cognitive_load', 'high_cognitive_load']:
        label_data = df[df['label'] == label]
        
        if len(label_data) < 125:  # Mínimo 0.5 segundos
            continue
        
        fz = label_data['channel_3'].values
        pz = label_data['channel_6'].values
        
        # Calcular PSD
        nperseg = min(len(fz), 250)
        freqs_fz, psd_fz = signal.welch(fz, 250, nperseg=nperseg, noverlap=nperseg//2)
        freqs_pz, psd_pz = signal.welch(pz, 250, nperseg=nperseg, noverlap=nperseg//2)
        
        # Bandas de frecuencia
        idx_theta = np.logical_and(freqs_fz >= 4.0, freqs_fz <= 7.0)
        idx_alpha = np.logical_and(freqs_pz >= 8.0, freqs_pz <= 12.0)
        
        # Potencias
        theta_power = np.trapz(psd_fz[idx_theta], freqs_fz[idx_theta]) if np.sum(idx_theta) > 0 else 0
        alpha_power = np.trapz(psd_pz[idx_alpha], freqs_pz[idx_alpha]) if np.sum(idx_alpha) > 0 else 0
        
        # Potencia total en bandas relevantes
        idx_wide_fz = np.logical_and(freqs_fz >= 1.0, freqs_fz <= 40.0)
        idx_wide_pz = np.logical_and(freqs_pz >= 1.0, freqs_pz <= 40.0)
        total_power_fz = np.trapz(psd_fz[idx_wide_fz], freqs_fz[idx_wide_fz]) if np.sum(idx_wide_fz) > 0 else 0
        total_power_pz = np.trapz(psd_pz[idx_wide_pz], freqs_pz[idx_wide_pz]) if np.sum(idx_wide_pz) > 0 else 0
        
        # Ratio
        ratio = theta_power / alpha_power if alpha_power > 0 else 0
        
        # Porcentaje de potencia en bandas
        theta_percent_fz = (theta_power / total_power_fz * 100) if total_power_fz > 0 else 0
        alpha_percent_pz = (alpha_power / total_power_pz * 100) if total_power_pz > 0 else 0
        
        results[label] = {
            'theta_power': theta_power,
            'alpha_power': alpha_power,
            'ratio': ratio,
            'theta_percent_fz': theta_percent_fz,
            'alpha_percent_pz': alpha_percent_pz,
            'total_power_fz': total_power_fz,
            'total_power_pz': total_power_pz
        }
    
    return subject, results

def main():
    """Análisis detallado."""
    print("="*70)
    print("ANALISIS DETALLADO: EDGAR Y JOSS vs OTROS SUJETOS")
    print("="*70)
    
    csv_files = {
        'Daniel': 'data_Daniel/eeg_data_20260120_124018.csv',
        'Edgar': 'data_Edgar/eeg_data_20260120_140754.csv',
        'Edwin': 'data_Edwin/eeg_data_20260120_130753.csv',
        'Jeronimo': 'data_Jeronimo/eeg_data_20260120_133259.csv',
        'Joss': 'data_Joss/eeg_data_20260120_120639.csv'
    }
    
    print("\n1. CALIDAD DE SEÑAL Y ESTADISTICAS BASICAS")
    print("="*70)
    
    signal_quality = {}
    for subject, csv_path in csv_files.items():
        subj, results = analyze_signal_quality(csv_path)
        signal_quality[subj] = results
        
        print(f"\n{subject}:")
        for label in ['low_cognitive_load', 'high_cognitive_load']:
            if label in results:
                r = results[label]
                print(f"  {label}:")
                print(f"    Muestras: {r['n_samples']}")
                print(f"    Fz: mean={r['fz_mean']:.2f}, std={r['fz_std']:.2f}, range={r['fz_range']:.2f}")
                print(f"    Pz: mean={r['pz_mean']:.2f}, std={r['pz_std']:.2f}, range={r['pz_range']:.2f}")
    
    print("\n2. CARACTERISTICAS ESPECTRALES")
    print("="*70)
    
    spectral_features = {}
    for subject, csv_path in csv_files.items():
        subj, results = calculate_spectral_features(csv_path)
        spectral_features[subj] = results
        
        print(f"\n{subject}:")
        for label in ['low_cognitive_load', 'high_cognitive_load']:
            if label in results:
                r = results[label]
                print(f"  {label}:")
                print(f"    Theta Power (Fz): {r['theta_power']:.3f} ({r['theta_percent_fz']:.1f}% del total)")
                print(f"    Alpha Power (Pz): {r['alpha_power']:.3f} ({r['alpha_percent_pz']:.1f}% del total)")
                print(f"    Ratio (Theta/Alpha): {r['ratio']:.3f}")
                print(f"    Total Power Fz: {r['total_power_fz']:.3f}")
                print(f"    Total Power Pz: {r['total_power_pz']:.3f}")
    
    print("\n3. COMPARACION: CUMPLEN vs NO CUMPLEN")
    print("="*70)
    
    cumplen = ['Daniel', 'Edwin', 'Jeronimo']
    no_cumplen = ['Edgar', 'Joss']
    
    print("\nSujetos que CUMPLEN (High > Low):")
    for subject in cumplen:
        if subject in spectral_features:
            low = spectral_features[subject]['low_cognitive_load']
            high = spectral_features[subject]['high_cognitive_load']
            print(f"\n  {subject}:")
            print(f"    Low:  Theta={low['theta_power']:.3f}, Alpha={low['alpha_power']:.3f}, Ratio={low['ratio']:.3f}")
            print(f"    High: Theta={high['theta_power']:.3f}, Alpha={high['alpha_power']:.3f}, Ratio={high['ratio']:.3f}")
            print(f"    Cambio Theta: {high['theta_power'] - low['theta_power']:.3f}")
            print(f"    Cambio Alpha: {high['alpha_power'] - low['alpha_power']:.3f}")
    
    print("\nSujetos que NO CUMPLEN (High <= Low):")
    for subject in no_cumplen:
        if subject in spectral_features:
            low = spectral_features[subject]['low_cognitive_load']
            high = spectral_features[subject]['high_cognitive_load']
            print(f"\n  {subject}:")
            print(f"    Low:  Theta={low['theta_power']:.3f}, Alpha={low['alpha_power']:.3f}, Ratio={low['ratio']:.3f}")
            print(f"    High: Theta={high['theta_power']:.3f}, Alpha={high['alpha_power']:.3f}, Ratio={high['ratio']:.3f}")
            print(f"    Cambio Theta: {high['theta_power'] - low['theta_power']:.3f}")
            print(f"    Cambio Alpha: {high['alpha_power'] - low['alpha_power']:.3f}")
    
    print("\n4. PATRONES OBSERVADOS")
    print("="*70)
    
    print("\nEn sujetos que CUMPLEN:")
    print("  - Theta generalmente AUMENTA de Low a High")
    print("  - Alpha generalmente DISMINUYE o se mantiene de Low a High")
    print("  - Resultado: Ratio aumenta")
    
    print("\nEn sujetos que NO CUMPLEN:")
    print("  - Revisar cambios específicos en Theta y Alpha")
    
    # Análisis específico
    print("\n5. ANALISIS ESPECIFICO DE EDGAR Y JOSS")
    print("="*70)
    
    for subject in no_cumplen:
        if subject in spectral_features:
            low = spectral_features[subject]['low_cognitive_load']
            high = spectral_features[subject]['high_cognitive_load']
            
            print(f"\n{subject}:")
            theta_change = high['theta_power'] - low['theta_power']
            alpha_change = high['alpha_power'] - low['alpha_power']
            
            print(f"  Cambio en Theta: {theta_change:.3f} ({theta_change/low['theta_power']*100:.1f}%)")
            print(f"  Cambio en Alpha: {alpha_change:.3f} ({alpha_change/low['alpha_power']*100:.1f}%)")
            
            if alpha_change > 0 and abs(alpha_change) > abs(theta_change):
                print(f"  [HALLAZGO] Alpha AUMENTA mas que Theta en High Load")
                print(f"  [HALLAZGO] Esto causa que el ratio disminuya")
            elif theta_change < 0:
                print(f"  [HALLAZGO] Theta DISMINUYE en High Load (inesperado)")
    
    print("\n" + "="*70)

if __name__ == '__main__':
    main()
