"""
Paso 2: Diagnóstico del problema de adquisición
Analiza por qué hay tan pocos datos guardados
"""

import pandas as pd
import numpy as np
from pathlib import Path
import os

# Configuración
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, 'data', 'Data-Experimento-Rafa')

# Configuración esperada
EXPECTED_DURATIONS = {
    'baseline_eyes_open': 90,  # segundos
    'baseline_eyes_closed': 90,  # segundos
    'low_cognitive_load': 180,  # 3 minutos
    'high_cognitive_load': 180  # 3 minutos
}

# Frecuencia de muestreo efectiva (con subsampling de 1 cada 5)
EFFECTIVE_SAMPLE_RATE = 250 / 5  # 50 Hz (porque se guarda cada 5 muestras)

def analyze_acquisition_issue(csv_path):
    """Analiza un archivo CSV para diagnosticar problemas de adquisición."""
    df = pd.read_csv(csv_path)
    subject_name = Path(csv_path).parent.name.replace('data_', '')
    
    print(f"\n{'='*80}")
    print(f"DIAGNÓSTICO: {subject_name}")
    print(f"{'='*80}")
    
    # 1. Análisis de duración por fase
    print("\n1. DURACIÓN REAL vs ESPERADA:")
    print("-" * 80)
    print(f"{'Fase':<30} {'Esperada (seg)':<18} {'Real (seg)':<15} {'Muestras':<12} {'% Esperado':<12}")
    print("-" * 80)
    
    issues = []
    for phase, expected_duration in EXPECTED_DURATIONS.items():
        phase_data = df[df['label'] == phase]
        actual_samples = len(phase_data)
        actual_duration = actual_samples / EFFECTIVE_SAMPLE_RATE
        expected_samples = expected_duration * EFFECTIVE_SAMPLE_RATE
        percentage = (actual_duration / expected_duration * 100) if expected_duration > 0 else 0
        
        print(f"{phase:<30} {expected_duration:<18} {actual_duration:<15.2f} {actual_samples:<12} {percentage:<12.1f}%")
        
        if percentage < 10:  # Menos del 10% de lo esperado
            issues.append(f"{phase}: Solo {percentage:.1f}% de la duración esperada")
    
    # 2. Análisis temporal
    print("\n2. ANÁLISIS TEMPORAL:")
    print("-" * 80)
    
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')
        total_time_span = df['timestamp'].max() - df['timestamp'].min()
        print(f"Rango temporal total: {total_time_span:.2f} segundos")
        print(f"Duración esperada total: {sum(EXPECTED_DURATIONS.values()):.2f} segundos")
        print(f"Ratio: {total_time_span / sum(EXPECTED_DURATIONS.values()) * 100:.1f}%")
        
        # Verificar si hay gaps en el tiempo
        df_sorted = df.sort_values('timestamp')
        time_diffs = df_sorted['timestamp'].diff().dropna()
        avg_interval = time_diffs.mean()
        expected_interval = 1.0 / EFFECTIVE_SAMPLE_RATE  # 0.02 segundos a 50 Hz
        
        print(f"\nIntervalo promedio entre muestras: {avg_interval:.4f} segundos")
        print(f"Intervalo esperado (50 Hz): {expected_interval:.4f} segundos")
        
        if avg_interval > expected_interval * 2:
            issues.append(f"Intervalos irregulares: promedio {avg_interval:.4f}s vs esperado {expected_interval:.4f}s")
    
    # 3. Análisis de fases
    print("\n3. SECUENCIA DE FASES:")
    print("-" * 80)
    
    if 'timestamp' in df.columns:
        df_sorted = df.sort_values('timestamp')
        phase_sequence = df_sorted['label'].unique()
        print("Fases en orden temporal:")
        for i, phase in enumerate(phase_sequence, 1):
            count = len(df_sorted[df_sorted['label'] == phase])
            print(f"  {i}. {phase} ({count} muestras)")
    
    # 4. Verificar si el experimento se completó
    print("\n4. ESTADO DEL EXPERIMENTO:")
    print("-" * 80)
    
    has_completed = 'completed' in df['label'].values if 'label' in df.columns else False
    has_analysis = 'analysis' in df['label'].values if 'label' in df.columns else False
    
    print(f"Fase 'completed' presente: {has_completed}")
    print(f"Fase 'analysis' presente: {has_analysis}")
    
    if not has_completed:
        issues.append("El experimento no parece haberse completado (falta fase 'completed')")
    
    # 5. Resumen de problemas
    print("\n5. PROBLEMAS IDENTIFICADOS:")
    print("-" * 80)
    
    if issues:
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")
    else:
        print("  ✓ No se identificaron problemas evidentes en la estructura")
    
    # 6. Hipótesis sobre el problema
    print("\n6. HIPÓTESIS SOBRE LA CAUSA:")
    print("-" * 80)
    
    # Calcular duración total real
    total_real_duration = len(df) / EFFECTIVE_SAMPLE_RATE
    total_expected = sum(EXPECTED_DURATIONS.values())
    
    if total_real_duration < total_expected * 0.1:
        print("  -> El experimento parece haberse detenido prematuramente")
        print("  -> Posibles causas:")
        print("     - El usuario guardo los datos antes de completar todas las fases")
        print("     - Hubo un error que detuvo el logging")
        print("     - El experimento se interrumpio manualmente")
    elif total_real_duration < total_expected * 0.5:
        print("  -> El experimento se completo parcialmente")
        print("  -> Posibles causas:")
        print("     - Algunas fases no se ejecutaron completamente")
        print("     - El logging se detuvo antes de tiempo")
    else:
        print("  -> La duracion parece razonable, pero las fases individuales son cortas")
        print("  -> Posible causa:")
        print("     - El subsampling esta funcionando, pero las fases se ejecutaron muy rapido")
    
    return {
        'subject': subject_name,
        'total_samples': len(df),
        'total_duration': total_real_duration,
        'expected_duration': total_expected,
        'percentage': total_real_duration / total_expected * 100 if total_expected > 0 else 0,
        'issues': issues
    }

def main():
    """Función principal."""
    print("="*80)
    print("PASO 2: DIAGNÓSTICO DEL PROBLEMA DE ADQUISICIÓN")
    print("="*80)
    print("\nAnalizando por qué hay tan pocos datos guardados...")
    print(f"\nConfiguración esperada:")
    print(f"  - Frecuencia de muestreo original: 250 Hz")
    print(f"  - Subsampling: 1 cada 5 muestras")
    print(f"  - Frecuencia efectiva guardada: {EFFECTIVE_SAMPLE_RATE} Hz")
    print(f"  - Duraciones esperadas:")
    for phase, duration in EXPECTED_DURATIONS.items():
        print(f"    * {phase}: {duration} segundos ({duration * EFFECTIVE_SAMPLE_RATE:.0f} muestras esperadas)")
    
    # Buscar archivos CSV
    csv_files = []
    data_path = Path(DATA_DIR)
    
    if data_path.exists():
        for data_dir in data_path.glob('data_*'):
            csv_files.extend(data_dir.glob('eeg_data_*.csv'))
    
    if not csv_files:
        print("\nNo se encontraron archivos CSV")
        return
    
    # Analizar cada archivo
    all_results = []
    for csv_file in sorted(csv_files):
        result = analyze_acquisition_issue(csv_file)
        all_results.append(result)
    
    # Resumen general
    print("\n" + "="*80)
    print("RESUMEN GENERAL")
    print("="*80)
    
    print(f"\n{'Sujeto':<15} {'Duración Real':<18} {'Duración Esperada':<20} {'%':<10} {'Problemas':<10}")
    print("-" * 80)
    
    for result in all_results:
        problems_count = len(result['issues'])
        print(f"{result['subject']:<15} {result['total_duration']:<18.2f} {result['expected_duration']:<20.2f} {result['percentage']:<10.1f} {problems_count:<10}")
    
    print("\n" + "="*80)
    print("CONCLUSIÓN")
    print("="*80)
    
    avg_percentage = np.mean([r['percentage'] for r in all_results])
    print(f"\nPromedio de duración capturada: {avg_percentage:.1f}% de lo esperado")
    
    if avg_percentage < 5:
        print("\n⚠ PROBLEMA CRÍTICO: Solo se capturó menos del 5% de los datos esperados")
        print("\nCausas más probables:")
        print("  1. Los datos se guardaron prematuramente (antes de completar las fases)")
        print("  2. El logging se detuvo por algún error")
        print("  3. El experimento se interrumpió manualmente")
        print("\nRecomendación: Revisar el código de logging y verificar si hay algún")
        print("mecanismo que detenga el logging antes de tiempo.")
    elif avg_percentage < 20:
        print("\n⚠ PROBLEMA MODERADO: Solo se capturó menos del 20% de los datos esperados")
        print("\nPosibles causas:")
        print("  1. Las fases se ejecutaron más rápido de lo esperado")
        print("  2. El subsampling está funcionando pero hay menos datos de los esperados")
    else:
        print("\n✓ La duración capturada parece razonable")

if __name__ == '__main__':
    main()
