"""
Script para analizar cognitive load aplicando WAAF (Wavelet-Assisted Adaptive Filter)
a todos los sujetos y comparar resultados antes y después del post-procesamiento.
"""

import pandas as pd
import numpy as np
from scipy import signal
from pathlib import Path
import glob
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from core.waaf_filter import waaf_filter_2d, HAS_WAVELETS

# Configuración
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'output', 'analysis_output')
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
    """
    Aplica Common Average Reference (CAR).
    
    Args:
        eeg_matrix: Array 2D [n_samples, n_channels]
        bad_channel_idx: Índice del canal defectuoso a excluir (None si no hay)
    """
    n_samples, n_channels = eeg_matrix.shape
    
    # Identificar canales válidos
    if bad_channel_idx is not None:
        valid_channels = [i for i in range(n_channels) if i != bad_channel_idx]
    else:
        valid_channels = list(range(n_channels))
    
    if len(valid_channels) == 0:
        return eeg_matrix
    
    # Calcular promedio de canales válidos
    car_reference = np.mean(eeg_matrix[:, valid_channels], axis=1, keepdims=True)
    
    # Restar el promedio a todos los canales
    eeg_car = eeg_matrix - car_reference
    
    return eeg_car

def calculate_bandpower(signal_data, freq_band, sample_rate):
    """Calcula la potencia espectral en una banda de frecuencias."""
    if len(signal_data) < sample_rate // 2:
        return 0.0
    
    nperseg = min(len(signal_data), sample_rate)
    freqs, psd = signal.welch(signal_data, sample_rate, nperseg=nperseg, noverlap=nperseg//2)
    
    idx_band = np.logical_and(freqs >= freq_band[0], freqs <= freq_band[1])
    
    if np.sum(idx_band) == 0:
        return 0.0
    
    bandpower = np.trapz(psd[idx_band], freqs[idx_band])
    return bandpower

def preprocess_with_waaf(df, label, apply_waaf=True):
    """
    Preprocesa datos de una fase específica con o sin WAAF.
    
    Args:
        df: DataFrame completo
        label: Label de la fase a procesar
        apply_waaf: Si True, aplica WAAF; si False, solo filtros básicos
        
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
        # Notch 60 Hz
        eeg_matrix[:, ch] = apply_notch_filter(eeg_matrix[:, ch], 60.0, SAMPLE_RATE)
        # Bandpass 1-40 Hz
        eeg_matrix[:, ch] = apply_bandpass_filter(eeg_matrix[:, ch], 1.0, 40.0, SAMPLE_RATE)
    
    # Paso 2: CAR
    eeg_matrix = apply_car(eeg_matrix, bad_channel_idx=None)
    
    # Paso 3: WAAF (opcional)
    if apply_waaf and HAS_WAVELETS:
        try:
            eeg_matrix, _, _ = waaf_filter_2d(eeg_matrix, wavelet='db4', level=7, attn=[1,2,3,4])
        except Exception as e:
            print(f"    Advertencia: Error aplicando WAAF: {e}")
    
    # Extraer Fz y Pz
    fz_clean = eeg_matrix[:, FZ_CHANNEL]
    pz_clean = eeg_matrix[:, PZ_CHANNEL]
    
    return fz_clean, pz_clean

def calculate_cognitive_load_by_phase(df, labels_of_interest, apply_waaf=True):
    """Calcula la carga cognitiva con o sin WAAF."""
    results = {}
    min_samples = SAMPLE_RATE // 2
    
    for label in labels_of_interest:
        label_data = df[df['label'] == label].copy()
        
        if len(label_data) < min_samples:
            results[label] = []
            continue
        
        # Preprocesar
        fz_clean, pz_clean = preprocess_with_waaf(df, label, apply_waaf=apply_waaf)
        
        if fz_clean is None or pz_clean is None:
            results[label] = []
            continue
        
        ratios = []
        
        # Ajustar tamaño de ventana
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
                # Ajustar filtro de cordura: permitir ratios más bajos (0.01 en lugar de 0.1)
                if np.isfinite(ratio) and 0.01 < ratio < 100:
                    ratios.append(ratio)
        
        results[label] = ratios
    
    return results

def analyze_subject(csv_path, apply_waaf=True):
    """Analiza un sujeto con o sin WAAF."""
    try:
        df = pd.read_csv(csv_path)
        subject_name = Path(csv_path).parent.name.replace('data_', '')
        
        # Calcular ratios
        ratios = calculate_cognitive_load_by_phase(df, LABELS_OF_INTEREST, apply_waaf=apply_waaf)
        
        # Extraer estadísticas
        stats = {}
        for label in LABELS_OF_INTEREST:
            if label in ratios and len(ratios[label]) > 0:
                stats[label] = {
                    'mean': np.mean(ratios[label]),
                    'median': np.median(ratios[label]),
                    'std': np.std(ratios[label]),
                    'n': len(ratios[label])
                }
            else:
                stats[label] = None
        
        # Verificar hipótesis
        hypothesis_met = False
        if 'high_cognitive_load' in stats and 'low_cognitive_load' in stats:
            if stats['high_cognitive_load'] and stats['low_cognitive_load']:
                high_mean = stats['high_cognitive_load']['mean']
                low_mean = stats['low_cognitive_load']['mean']
                hypothesis_met = high_mean > low_mean
        
        return {
            'subject': subject_name,
            'stats': stats,
            'hypothesis_met': hypothesis_met
        }
    except Exception as e:
        print(f"Error analizando {csv_path}: {str(e)}")
        return None

def main():
    """Función principal."""
    if not HAS_WAVELETS:
        print("ERROR: PyWavelets no esta instalado.")
        print("Instalar con: pip install PyWavelets")
        return
    
    print("="*70)
    print("ANALISIS CON WAAF (Wavelet-Assisted Adaptive Filter)")
    print("="*70)
    
    # Encontrar todos los archivos CSV
    csv_files = []
    for data_dir in Path('.').glob('data_*'):
        csv_files.extend(data_dir.glob('eeg_data_*.csv'))
    
    if not csv_files:
        print("No se encontraron archivos CSV")
        return
    
    print(f"\nEncontrados {len(csv_files)} archivos CSV")
    
    # Analizar sin WAAF
    print("\n" + "="*70)
    print("ANALISIS SIN WAAF")
    print("="*70)
    
    results_without = {}
    for csv_file in sorted(csv_files):
        result = analyze_subject(csv_file, apply_waaf=False)
        if result:
            results_without[result['subject']] = result
    
    # Analizar con WAAF
    print("\n" + "="*70)
    print("ANALISIS CON WAAF")
    print("="*70)
    
    results_with = {}
    for csv_file in sorted(csv_files):
        result = analyze_subject(csv_file, apply_waaf=True)
        if result:
            results_with[result['subject']] = result
    
    # Comparar resultados
    print("\n" + "="*70)
    print("COMPARACION: SIN WAAF vs CON WAAF")
    print("="*70)
    
    print(f"\n{'Sujeto':<15} {'Sin WAAF':<25} {'Con WAAF':<25} {'Mejora':<15}")
    print("-" * 80)
    
    # Mostrar todos los sujetos encontrados
    all_subjects = set(results_without.keys()) | set(results_with.keys())
    
    for subject in sorted(all_subjects):
        if subject not in results_without or subject not in results_with:
            print(f"{subject:<15} {'ERROR: Datos incompletos'}")
            continue
        
        without = results_without[subject]
        with_waaf = results_with[subject]
        
        # Obtener ratios High y Low
        if (without['stats']['high_cognitive_load'] and 
            without['stats']['low_cognitive_load'] and
            with_waaf['stats']['high_cognitive_load'] and
            with_waaf['stats']['low_cognitive_load']):
            
            high_without = without['stats']['high_cognitive_load']['mean']
            low_without = without['stats']['low_cognitive_load']['mean']
            diff_without = high_without - low_without
            hyp_without = "CUMPLE" if without['hypothesis_met'] else "NO"
            
            high_with = with_waaf['stats']['high_cognitive_load']['mean']
            low_with = with_waaf['stats']['low_cognitive_load']['mean']
            diff_with = high_with - low_with
            hyp_with = "CUMPLE" if with_waaf['hypothesis_met'] else "NO"
            
            mejora = ""
            if not without['hypothesis_met'] and with_waaf['hypothesis_met']:
                mejora = "MEJORO"
            elif without['hypothesis_met'] and not with_waaf['hypothesis_met']:
                mejora = "EMPEORO"
            elif diff_with > diff_without:
                mejora = "MEJORO"
            elif diff_with < diff_without:
                mejora = "EMPEORO"
            else:
                mejora = "IGUAL"
            
            print(f"{subject:<15} High={high_without:.3f} Low={low_without:.3f} {hyp_without:<5} | "
                  f"High={high_with:.3f} Low={low_with:.3f} {hyp_with:<5} | {mejora}")
        else:
            # Mostrar si falta algún dato
            missing = []
            if not without['stats']['high_cognitive_load']:
                missing.append('High(sin)')
            if not without['stats']['low_cognitive_load']:
                missing.append('Low(sin)')
            if not with_waaf['stats']['high_cognitive_load']:
                missing.append('High(con)')
            if not with_waaf['stats']['low_cognitive_load']:
                missing.append('Low(con)')
            print(f"{subject:<15} {'ERROR: Faltan datos - ' + ', '.join(missing)}")
    
    # Resumen detallado
    print("\n" + "="*70)
    print("RESUMEN DETALLADO")
    print("="*70)
    
    cumplen_sin = sum(1 for r in results_without.values() if r.get('hypothesis_met', False))
    cumplen_con = sum(1 for r in results_with.values() if r.get('hypothesis_met', False))
    
    print(f"\nSujetos que cumplen hipotesis (High > Low):")
    print(f"  Sin WAAF: {cumplen_sin}/{len(results_without)}")
    print(f"  Con WAAF: {cumplen_con}/{len(results_with)}")
    
    # Análisis de cambios en diferencias
    print(f"\nCambios en diferencias (High - Low):")
    for subject in sorted(all_subjects):
        if subject in results_without and subject in results_with:
            without = results_without[subject]
            with_waaf = results_with[subject]
            
            if (without['stats']['high_cognitive_load'] and 
                without['stats']['low_cognitive_load'] and
                with_waaf['stats']['high_cognitive_load'] and
                with_waaf['stats']['low_cognitive_load']):
                
                diff_without = without['stats']['high_cognitive_load']['mean'] - without['stats']['low_cognitive_load']['mean']
                diff_with = with_waaf['stats']['high_cognitive_load']['mean'] - with_waaf['stats']['low_cognitive_load']['mean']
                cambio = diff_with - diff_without
                
                print(f"  {subject}: {diff_without:.3f} -> {diff_with:.3f} (cambio: {cambio:+.3f})")
    
    if cumplen_con > cumplen_sin:
        print(f"\n[RESULTADO] WAAF MEJORA los resultados: {cumplen_con - cumplen_sin} sujetos adicionales cumplen la hipotesis")
    elif cumplen_con < cumplen_sin:
        print(f"\n[RESULTADO] WAAF EMPEORA los resultados: {cumplen_sin - cumplen_con} sujetos menos cumplen la hipotesis")
    else:
        print(f"\n[RESULTADO] WAAF no cambia el numero de sujetos que cumplen")
        print(f"  Sin embargo, hay cambios en los valores de los ratios")
        print(f"  Revisar si las diferencias (High - Low) aumentan o disminuyen")
    
    print("\n" + "="*70)

if __name__ == '__main__':
    main()
