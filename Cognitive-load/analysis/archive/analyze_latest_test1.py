"""
Análisis completo del experimento más reciente en data_test1
"""

import pandas as pd
import numpy as np
from pathlib import Path
import os

# Configuración
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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
    print("ERROR: No se encontraron archivos CSV en data_test1")
    exit(1)

CSV_PATH = str(CSV_PATH)

def analyze_latest():
    """Analiza el experimento más reciente."""
    print("="*80)
    print("ANALISIS DEL EXPERIMENTO MAS RECIENTE - data_test1")
    print("="*80)
    print(f"\nArchivo: {Path(CSV_PATH).name}")
    
    # Cargar datos
    print("\nCargando datos...")
    df = pd.read_csv(CSV_PATH)
    print(f"Total de muestras: {len(df):,}")
    
    # Información básica
    print(f"\n1. INFORMACION BASICA:")
    print("-"*80)
    print(f"Columnas: {list(df.columns)}")
    
    # Análisis temporal
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')
        df_sorted = df.sort_values('timestamp')
        
        total_time_span = df_sorted['timestamp'].max() - df_sorted['timestamp'].min()
        total_duration_min = total_time_span / 60.0
        
        print(f"\nRango temporal: {total_time_span:.2f} segundos ({total_duration_min:.2f} minutos)")
        
        # Calcular frecuencia
        time_diffs = df_sorted['timestamp'].diff().dropna()
        median_interval = time_diffs.median()
        mean_interval = time_diffs.mean()
        
        if median_interval > 0:
            freq_median = 1.0 / median_interval
            freq_mean = 1.0 / mean_interval
            print(f"Intervalo mediano: {median_interval:.6f} seg")
            print(f"Intervalo promedio: {mean_interval:.6f} seg")
            print(f"Frecuencia (mediana): {freq_median:.2f} Hz")
            print(f"Frecuencia (promedio): {freq_mean:.2f} Hz")
            
            # Verificar subsampling
            expected_50hz = 1.0 / 50.0
            expected_250hz = 1.0 / 250.0
            ratio_50hz = median_interval / expected_50hz
            ratio_250hz = median_interval / expected_250hz
            
            print(f"\nVerificacion de subsampling:")
            print(f"  Ratio vs 50 Hz esperado: {ratio_50hz:.3f}x")
            print(f"  Ratio vs 250 Hz esperado: {ratio_250hz:.3f}x")
            
            if 0.8 < ratio_50hz < 1.2:
                print(f"  [OK] Subsampling funcionando correctamente (~50 Hz)")
            elif 0.8 < ratio_250hz < 1.2:
                print(f"  [ADVERTENCIA] No hay subsampling (~250 Hz)")
            else:
                print(f"  [PROBLEMA] Frecuencia inconsistente")
    
    # Análisis por fase
    print(f"\n2. ANALISIS POR FASE:")
    print("-"*80)
    
    if 'label' in df.columns:
        phase_counts = df['label'].value_counts()
        print(f"\nFases encontradas ({len(phase_counts)}):")
        
        # Duraciones esperadas
        expected_durations = {
            'baseline_eyes_open': 90,
            'baseline_eyes_closed': 90,
            'low_cognitive_load': 180,
            'high_cognitive_load': 180
        }
        
        print(f"\n{'Fase':<30} {'Muestras':<12} {'Duracion (seg)':<18} {'Esperada (seg)':<18} {'%':<10} {'Estado':<15}")
        print("-"*100)
        
        for phase in phase_counts.index:
            phase_data = df[df['label'] == phase].sort_values('timestamp')
            count = len(phase_data)
            
            if len(phase_data) > 1 and 'timestamp' in df.columns:
                time_span = phase_data['timestamp'].max() - phase_data['timestamp'].min()
                if median_interval > 0:
                    calculated_duration = count / freq_median
                else:
                    calculated_duration = time_span
            else:
                time_span = 0
                calculated_duration = 0
            
            expected = expected_durations.get(phase, 0)
            if expected > 0:
                percentage = (time_span / expected * 100) if expected > 0 else 0
                if 90 < percentage < 110:
                    status = "OK"
                elif percentage < 50:
                    status = "MUY CORTO"
                elif percentage > 200:
                    status = "MUY LARGO"
                else:
                    status = "IRREGULAR"
            else:
                percentage = 0
                status = "N/A"
            
            print(f"{phase:<30} {count:<12,} {time_span:<18.1f} {expected:<18} {percentage:<10.1f}% {status:<15}")
    
    # Verificar gaps
    print(f"\n3. VERIFICACION DE GAPS:")
    print("-"*80)
    
    if 'timestamp' in df.columns:
        time_diffs = df_sorted['timestamp'].diff().dropna()
        normal_intervals = time_diffs[time_diffs < 1.0]
        outlier_intervals = time_diffs[time_diffs >= 1.0]
        
        print(f"Intervalos normales (< 1 seg): {len(normal_intervals):,} ({len(normal_intervals)/len(time_diffs)*100:.1f}%)")
        print(f"Gaps (>= 1 seg): {len(outlier_intervals):,} ({len(outlier_intervals)/len(time_diffs)*100:.1f}%)")
        
        if len(outlier_intervals) > 0:
            print(f"\nGaps detectados:")
            print(f"  Numero: {len(outlier_intervals)}")
            print(f"  Promedio: {outlier_intervals.mean():.2f} seg")
            print(f"  Maximo: {outlier_intervals.max():.2f} seg")
        else:
            print("\n[OK] No se detectaron gaps grandes")
    
    # Estadísticas de canales
    print(f"\n4. ESTADISTICAS DE CANALES:")
    print("-"*80)
    
    channel_cols = [col for col in df.columns if 'channel_' in col]
    print(f"\nCanales: {len(channel_cols)}")
    
    for ch_col in channel_cols[:3]:  # Primeros 3
        ch_data = df[ch_col].dropna()
        if len(ch_data) > 0:
            print(f"\n  {ch_col}:")
            print(f"    Media: {ch_data.mean():.2f}")
            print(f"    Std: {ch_data.std():.2f}")
            print(f"    Min: {ch_data.min():.2f}")
            print(f"    Max: {ch_data.max():.2f}")
            print(f"    NaN: {df[ch_col].isna().sum()}")
    
    # Comparación con experimento anterior
    print(f"\n5. COMPARACION CON EXPERIMENTO ANTERIOR:")
    print("-"*80)
    
    old_samples = 260744
    old_freq = 415.0
    old_duration = 797.32
    
    improvement_samples = (len(df) / old_samples * 100) if old_samples > 0 else 0
    improvement_freq = ((freq_median / old_freq - 1) * 100) if old_freq > 0 else 0
    
    print(f"Experimento anterior:")
    print(f"  Muestras: {old_samples:,}")
    print(f"  Frecuencia: {old_freq:.1f} Hz")
    print(f"  Duracion: {old_duration:.1f} seg ({old_duration/60:.1f} min)")
    
    print(f"\nExperimento actual:")
    print(f"  Muestras: {len(df):,}")
    print(f"  Frecuencia: {freq_median:.1f} Hz")
    print(f"  Duracion: {total_time_span:.1f} seg ({total_duration_min:.1f} min)")
    
    print(f"\nCambios:")
    print(f"  Muestras: {improvement_samples:.1f}% del anterior")
    print(f"  Frecuencia: {improvement_freq:+.1f}% cambio")
    
    # Conclusiones
    print(f"\n6. CONCLUSIONES:")
    print("-"*80)
    
    conclusions = []
    
    # Subsampling
    if 0.8 < ratio_50hz < 1.2:
        conclusions.append("[OK] Subsampling funcionando correctamente")
    else:
        conclusions.append("[PROBLEMA] Subsampling NO funcionando (frecuencia: {:.1f} Hz)".format(freq_median))
    
    # Gaps
    if len(outlier_intervals) == 0:
        conclusions.append("[OK] No hay gaps grandes en los datos")
    else:
        conclusions.append("[PROBLEMA] Se detectaron {} gaps grandes".format(len(outlier_intervals)))
    
    # Duraciones
    if 'label' in df.columns:
        phases_ok = 0
        phases_problem = 0
        for phase in expected_durations.keys():
            if phase in phase_counts.index:
                phase_data = df[df['label'] == phase].sort_values('timestamp')
                if len(phase_data) > 1:
                    time_span = phase_data['timestamp'].max() - phase_data['timestamp'].min()
                    expected = expected_durations[phase]
                    percentage = (time_span / expected * 100) if expected > 0 else 0
                    if 90 < percentage < 110:
                        phases_ok += 1
                    else:
                        phases_problem += 1
        
        if phases_problem == 0:
            conclusions.append("[OK] Todas las fases tienen duraciones correctas")
        else:
            conclusions.append("[PROBLEMA] {} fases con duraciones irregulares".format(phases_problem))
    
    for i, conclusion in enumerate(conclusions, 1):
        print(f"  {i}. {conclusion}")
    
    print("\n" + "="*80)
    print("ANALISIS COMPLETADO")
    print("="*80)

if __name__ == '__main__':
    analyze_latest()
