"""
Script para verificar que los timers de las fases experimentales funcionen correctamente
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

# Duraciones esperadas (en segundos)
EXPECTED_DURATIONS = {
    'baseline_eyes_open': 90,
    'baseline_eyes_closed': 90,
    'low_cognitive_load': 180,
    'high_cognitive_load': 180
}

def verify_timers(csv_path):
    """Verifica que las duraciones de las fases coincidan con los timers."""
    print("="*80)
    print("VERIFICACION DE TIMERS DE FASES EXPERIMENTALES")
    print("="*80)
    
    # Cargar datos
    df = pd.read_csv(csv_path)
    print(f"\nArchivo: {Path(csv_path).name}")
    
    if 'timestamp' not in df.columns or 'label' not in df.columns:
        print("\nERROR: Faltan columnas necesarias (timestamp o label)")
        return
    
    df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')
    
    # Calcular frecuencia de muestreo efectiva
    df_sorted = df.sort_values('timestamp')
    time_diffs = df_sorted['timestamp'].diff().dropna()
    median_interval = time_diffs.median()
    effective_sample_rate = 1.0 / median_interval if median_interval > 0 else 250
    
    print(f"\nFrecuencia de muestreo efectiva: {effective_sample_rate:.1f} Hz")
    
    # Analizar cada fase
    print("\n" + "="*80)
    print("ANALISIS POR FASE")
    print("="*80)
    
    print(f"\n{'Fase':<30} {'Esperada (seg)':<18} {'Real (seg)':<15} {'Muestras':<12} {'%':<10} {'Estado':<15}")
    print("-"*100)
    
    issues = []
    for phase, expected_duration in EXPECTED_DURATIONS.items():
        phase_data = df[df['label'] == phase].sort_values('timestamp')
        
        if len(phase_data) == 0:
            print(f"{phase:<30} {expected_duration:<18} {'0':<15} {'0':<12} {'0':<10}% {'SIN DATOS':<15}")
            issues.append(f"{phase}: Sin datos")
            continue
        
        # Calcular duración real
        time_span = phase_data['timestamp'].max() - phase_data['timestamp'].min()
        sample_count = len(phase_data)
        calculated_duration = sample_count / effective_sample_rate
        
        percentage = (time_span / expected_duration * 100) if expected_duration > 0 else 0
        
        # Determinar estado
        if 90 < percentage < 110:
            status = "OK"
        elif percentage < 50:
            status = "MUY CORTO"
            issues.append(f"{phase}: Solo {percentage:.1f}% de duracion esperada")
        elif percentage > 200:
            status = "MUY LARGO"
            issues.append(f"{phase}: {percentage:.1f}% de duracion esperada (timer no detuvo?)")
        else:
            status = "IRREGULAR"
            issues.append(f"{phase}: Duracion irregular ({percentage:.1f}%)")
        
        print(f"{phase:<30} {expected_duration:<18} {time_span:<15.1f} {sample_count:<12,} {percentage:<10.1f}% {status:<15}")
    
    # Verificar transiciones
    print("\n" + "="*80)
    print("VERIFICACION DE TRANSICIONES")
    print("="*80)
    
    # Ordenar por timestamp
    df_sorted = df.sort_values('timestamp')
    
    # Verificar secuencia de fases
    phase_sequence = []
    current_phase = None
    for idx, row in df_sorted.iterrows():
        if row['label'] != current_phase:
            if current_phase is not None:
                phase_sequence.append((current_phase, row['timestamp']))
            current_phase = row['label']
    
    print("\nSecuencia de fases detectada:")
    for i, (phase, timestamp) in enumerate(phase_sequence, 1):
        print(f"  {i}. {phase} (inicio: {timestamp:.2f})")
    
    # Verificar transiciones esperadas
    expected_sequence = [
        'setup',
        'baseline_eyes_open',
        'baseline_eyes_closed',
        'low_cognitive_load',
        'high_cognitive_load'
    ]
    
    detected_phases = [p[0] for p in phase_sequence]
    print("\nTransiciones esperadas vs detectadas:")
    for i, expected in enumerate(expected_sequence):
        if i < len(detected_phases):
            detected = detected_phases[i]
            match = "OK" if expected == detected else "NO COINCIDE"
            print(f"  {i+1}. Esperado: {expected:<25} Detectado: {detected:<25} {match}")
        else:
            print(f"  {i+1}. Esperado: {expected:<25} Detectado: {'NO ENCONTRADO':<25} FALTA")
    
    # Resumen de problemas
    print("\n" + "="*80)
    print("RESUMEN DE PROBLEMAS")
    print("="*80)
    
    if issues:
        print("\nProblemas detectados:")
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")
    else:
        print("\n[OK] No se detectaron problemas con los timers")
    
    print("\n" + "="*80)
    print("VERIFICACION COMPLETADA")
    print("="*80)

if __name__ == '__main__':
    verify_timers(CSV_PATH)
