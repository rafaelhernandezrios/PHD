"""
Análisis del experimento completo en data_test1
"""

import pandas as pd
import numpy as np
from pathlib import Path
import os

# Configuración
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(BASE_DIR, 'data_test1', 'eeg_data_20260121_225034.csv')

# Frecuencia efectiva (con subsampling de 1 cada 5)
EFFECTIVE_SAMPLE_RATE = 250 / 5  # 50 Hz

def analyze_test1():
    """Analiza el experimento completo."""
    print("="*80)
    print("ANÁLISIS DEL EXPERIMENTO COMPLETO - data_test1")
    print("="*80)
    
    # Cargar datos
    print("\nCargando datos...")
    df = pd.read_csv(CSV_PATH)
    
    print(f"\n1. INFORMACIÓN GENERAL:")
    print("-"*80)
    print(f"Total de muestras: {len(df):,}")
    print(f"Columnas: {list(df.columns)}")
    
    # Análisis temporal
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')
        total_time_span = df['timestamp'].max() - df['timestamp'].min()
        total_duration_min = total_time_span / 60.0
        
        print(f"\nRango temporal: {total_time_span:.2f} segundos ({total_duration_min:.2f} minutos)")
        print(f"Frecuencia efectiva: {EFFECTIVE_SAMPLE_RATE} Hz")
        print(f"Muestras esperadas: {total_time_span * EFFECTIVE_SAMPLE_RATE:.0f}")
        print(f"Muestras reales: {len(df):,}")
        
        # Verificar intervalos
        df_sorted = df.sort_values('timestamp')
        time_diffs = df_sorted['timestamp'].diff().dropna()
        avg_interval = time_diffs.mean()
        expected_interval = 1.0 / EFFECTIVE_SAMPLE_RATE
        
        print(f"\nIntervalo promedio: {avg_interval:.4f} segundos")
        print(f"Intervalo esperado (50 Hz): {expected_interval:.4f} segundos")
        print(f"Ratio: {avg_interval / expected_interval:.2f}x")
    
    # Análisis por fase
    print(f"\n2. ANÁLISIS POR FASE:")
    print("-"*80)
    
    if 'label' in df.columns:
        phase_counts = df['label'].value_counts()
        print(f"\nFases encontradas ({len(phase_counts)}):")
        
        phase_stats = []
        for phase, count in phase_counts.items():
            phase_data = df[df['label'] == phase]
            duration_sec = count / EFFECTIVE_SAMPLE_RATE
            duration_min = duration_sec / 60.0
            
            if 'timestamp' in df.columns:
                phase_time_span = phase_data['timestamp'].max() - phase_data['timestamp'].min()
            else:
                phase_time_span = duration_sec
            
            phase_stats.append({
                'phase': phase,
                'samples': count,
                'duration_sec': duration_sec,
                'duration_min': duration_min,
                'time_span': phase_time_span
            })
            
            print(f"\n  {phase}:")
            print(f"    Muestras: {count:,}")
            print(f"    Duración calculada: {duration_sec:.1f} seg ({duration_min:.2f} min)")
            print(f"    Rango temporal: {phase_time_span:.1f} seg")
        
        # Fases de interés
        print(f"\n3. FASES DE INTERÉS:")
        print("-"*80)
        
        phases_of_interest = [
            'baseline_eyes_open',
            'baseline_eyes_closed',
            'low_cognitive_load',
            'high_cognitive_load'
        ]
        
        expected_durations = {
            'baseline_eyes_open': 90,
            'baseline_eyes_closed': 90,
            'low_cognitive_load': 180,
            'high_cognitive_load': 180
        }
        
        print(f"\n{'Fase':<30} {'Esperada (seg)':<18} {'Real (seg)':<15} {'Muestras':<12} {'%':<10}")
        print("-"*80)
        
        for phase in phases_of_interest:
            if phase in phase_counts:
                count = phase_counts[phase]
                actual_duration = count / EFFECTIVE_SAMPLE_RATE
                expected_duration = expected_durations.get(phase, 0)
                percentage = (actual_duration / expected_duration * 100) if expected_duration > 0 else 0
                
                print(f"{phase:<30} {expected_duration:<18} {actual_duration:<15.1f} {count:<12,} {percentage:<10.1f}%")
            else:
                print(f"{phase:<30} {'N/A':<18} {'0':<15} {'0':<12} {'0':<10}%")
    
    # Estadísticas de canales
    print(f"\n4. ESTADÍSTICAS DE CANALES:")
    print("-"*80)
    
    channel_cols = [col for col in df.columns if 'channel_' in col]
    print(f"\nCanales encontrados: {len(channel_cols)}")
    
    for ch_col in channel_cols[:3]:  # Solo primeros 3 para no saturar
        ch_data = df[ch_col].dropna()
        if len(ch_data) > 0:
            print(f"\n  {ch_col}:")
            print(f"    Media: {ch_data.mean():.2f}")
            print(f"    Std: {ch_data.std():.2f}")
            print(f"    Min: {ch_data.min():.2f}")
            print(f"    Max: {ch_data.max():.2f}")
            print(f"    NaN: {df[ch_col].isna().sum()}")
    
    # Comparación con datos anteriores
    print(f"\n5. COMPARACIÓN CON DATOS ANTERIORES:")
    print("-"*80)
    
    old_avg_samples = 1600  # Promedio de datos anteriores
    improvement = (len(df) / old_avg_samples) if old_avg_samples > 0 else 0
    
    print(f"Muestras anteriores (promedio): {old_avg_samples:,}")
    print(f"Muestras actuales: {len(df):,}")
    print(f"Mejora: {improvement:.1f}x más datos")
    
    print("\n" + "="*80)
    print("ANÁLISIS COMPLETADO")
    print("="*80)

if __name__ == '__main__':
    analyze_test1()
