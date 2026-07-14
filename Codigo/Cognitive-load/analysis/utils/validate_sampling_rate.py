"""
Script para validar la frecuencia de muestreo real del dispositivo
y verificar el subsampling en los datos guardados
"""

import pandas as pd
import numpy as np
from pathlib import Path
import os

# Configuración
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, 'data_test1')

# Buscar el archivo CSV más reciente
def find_latest_csv(data_dir):
    """Encuentra el archivo CSV más reciente en el directorio."""
    csv_files = list(Path(data_dir).glob('*.csv'))
    if not csv_files:
        return None
    return max(csv_files, key=lambda p: p.stat().st_mtime)

CSV_PATH = find_latest_csv(DATA_DIR)
if CSV_PATH is None:
    CSV_PATH = os.path.join(DATA_DIR, 'eeg_data_20260121_225034.csv')  # Fallback
else:
    CSV_PATH = str(CSV_PATH)

def validate_sampling_rate(csv_path):
    """Valida la frecuencia de muestreo real en un archivo CSV."""
    print("="*80)
    print("VALIDACION DE FRECUENCIA DE MUESTREO")
    print("="*80)
    
    # Cargar datos
    df = pd.read_csv(csv_path)
    print(f"\nArchivo: {Path(csv_path).name}")
    print(f"Total de muestras: {len(df):,}")
    
    if 'timestamp' not in df.columns:
        print("\nERROR: No hay columna 'timestamp' en el archivo")
        return
    
    # Convertir timestamps a numérico
    df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')
    df = df.sort_values('timestamp')
    
    # Calcular intervalos entre muestras
    time_diffs = df['timestamp'].diff().dropna()
    
    # Estadísticas de intervalos
    print("\n1. ESTADISTICAS DE INTERVALOS:")
    print("-"*80)
    print(f"Intervalo minimo: {time_diffs.min():.6f} segundos")
    print(f"Intervalo maximo: {time_diffs.max():.6f} segundos")
    print(f"Intervalo promedio: {time_diffs.mean():.6f} segundos")
    print(f"Intervalo mediano: {time_diffs.median():.6f} segundos")
    print(f"Intervalo std: {time_diffs.std():.6f} segundos")
    
    # Frecuencias calculadas
    print("\n2. FRECUENCIAS CALCULADAS:")
    print("-"*80)
    median_interval = time_diffs.median()
    mean_interval = time_diffs.mean()
    
    if median_interval > 0:
        freq_from_median = 1.0 / median_interval
        print(f"Frecuencia desde mediana: {freq_from_median:.2f} Hz")
    
    if mean_interval > 0:
        freq_from_mean = 1.0 / mean_interval
        print(f"Frecuencia desde promedio: {freq_from_mean:.2f} Hz")
    
    # Frecuencias esperadas
    print("\n3. FRECUENCIAS ESPERADAS:")
    print("-"*80)
    print(f"Frecuencia original del dispositivo: 250 Hz")
    print(f"Intervalo esperado (250 Hz): {1.0/250:.6f} segundos")
    print(f"\nCon subsampling de 1 cada 5 (50 Hz):")
    print(f"Intervalo esperado (50 Hz): {1.0/50:.6f} segundos")
    
    # Análisis de subsampling
    print("\n4. ANALISIS DE SUBMUESTREO:")
    print("-"*80)
    expected_interval_50hz = 1.0 / 50.0
    expected_interval_250hz = 1.0 / 250.0
    
    if median_interval > 0:
        ratio_to_50hz = median_interval / expected_interval_50hz
        ratio_to_250hz = median_interval / expected_interval_250hz
        
        print(f"Ratio vs 50 Hz esperado: {ratio_to_50hz:.3f}x")
        print(f"Ratio vs 250 Hz esperado: {ratio_to_250hz:.3f}x")
        
        if 0.8 < ratio_to_50hz < 1.2:
            print("\n[OK] La frecuencia es consistente con 50 Hz (subsampling funcionando)")
        elif 0.8 < ratio_to_250hz < 1.2:
            print("\n[ADVERTENCIA] La frecuencia es consistente con 250 Hz (subsampling NO funcionando)")
        else:
            print(f"\n[PROBLEMA] La frecuencia no coincide con ninguna esperada")
            print(f"          Puede haber problemas de logging o timestamps")
    
    # Distribución de intervalos
    print("\n5. DISTRIBUCION DE INTERVALOS:")
    print("-"*80)
    # Filtrar outliers (intervalos > 1 segundo probablemente son gaps)
    normal_intervals = time_diffs[time_diffs < 1.0]
    outlier_intervals = time_diffs[time_diffs >= 1.0]
    
    print(f"Intervalos normales (< 1 seg): {len(normal_intervals):,} ({len(normal_intervals)/len(time_diffs)*100:.1f}%)")
    print(f"Intervalos anormales (>= 1 seg): {len(outlier_intervals):,} ({len(outlier_intervals)/len(time_diffs)*100:.1f}%)")
    
    if len(normal_intervals) > 0:
        print(f"\nEstadisticas de intervalos normales:")
        print(f"  Mediana: {normal_intervals.median():.6f} seg")
        print(f"  Frecuencia: {1.0/normal_intervals.median():.2f} Hz")
    
    if len(outlier_intervals) > 0:
        print(f"\nGaps detectados:")
        print(f"  Numero de gaps: {len(outlier_intervals)}")
        print(f"  Gap promedio: {outlier_intervals.mean():.2f} seg")
        print(f"  Gap maximo: {outlier_intervals.max():.2f} seg")
    
    # Verificar consistencia por fase
    print("\n6. CONSISTENCIA POR FASE:")
    print("-"*80)
    
    if 'label' in df.columns:
        phases = df['label'].unique()
        for phase in phases:
            phase_data = df[df['label'] == phase].sort_values('timestamp')
            if len(phase_data) > 1:
                phase_diffs = phase_data['timestamp'].diff().dropna()
                phase_median = phase_diffs.median()
                if phase_median > 0:
                    phase_freq = 1.0 / phase_median
                    print(f"{phase}: {phase_freq:.2f} Hz (mediana: {phase_median:.6f} seg)")
    
    print("\n" + "="*80)
    print("VALIDACION COMPLETADA")
    print("="*80)

if __name__ == '__main__':
    validate_sampling_rate(CSV_PATH)
