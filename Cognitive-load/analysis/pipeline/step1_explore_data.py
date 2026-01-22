"""
Paso 1: Exploración inicial de datos
- Lista todos los archivos CSV disponibles
- Muestra estadísticas básicas por sujeto
- Cuenta muestras por fase
- Identifica datos faltantes o problemáticos
"""

import pandas as pd
import numpy as np
from pathlib import Path
import os

# Configuración
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, 'data', 'Data-Experimento-Rafa')

# Labels de interés
LABELS_OF_INTEREST = [
    'baseline_eyes_open',
    'baseline_eyes_closed',
    'low_cognitive_load',
    'high_cognitive_load'
]

def explore_csv_file(csv_path):
    """Explora un archivo CSV y retorna estadísticas básicas."""
    try:
        df = pd.read_csv(csv_path)
        subject_name = Path(csv_path).parent.name.replace('data_', '')
        
        # Información básica
        total_samples = len(df)
        total_duration = total_samples / 250.0  # Asumiendo 250 Hz
        
        # Columnas disponibles
        columns = list(df.columns)
        has_channels = any('channel_' in col for col in columns)
        has_label = 'label' in columns
        has_phase = 'phase' in columns or 'phase_label' in columns
        
        # Contar muestras por fase/label
        label_counts = {}
        if has_label:
            label_counts = df['label'].value_counts().to_dict()
        elif has_phase:
            phase_col = 'phase' if 'phase' in columns else 'phase_label'
            label_counts = df[phase_col].value_counts().to_dict()
        
        # Verificar canales
        channel_cols = [col for col in columns if 'channel_' in col]
        n_channels = len(channel_cols)
        
        # Estadísticas de valores
        channel_stats = {}
        if channel_cols:
            for ch_col in channel_cols:
                ch_data = df[ch_col].dropna()
                if len(ch_data) > 0:
                    channel_stats[ch_col] = {
                        'mean': float(ch_data.mean()),
                        'std': float(ch_data.std()),
                        'min': float(ch_data.min()),
                        'max': float(ch_data.max()),
                        'n_nan': int(df[ch_col].isna().sum())
                    }
        
        # Verificar datos faltantes por fase
        missing_by_phase = {}
        if has_label:
            for label in LABELS_OF_INTEREST:
                label_data = df[df['label'] == label]
                if len(label_data) > 0:
                    missing_by_phase[label] = {
                        'samples': len(label_data),
                        'duration_sec': len(label_data) / 250.0,
                        'missing_channels': {}
                    }
                    for ch_col in channel_cols:
                        n_missing = label_data[ch_col].isna().sum()
                        if n_missing > 0:
                            missing_by_phase[label]['missing_channels'][ch_col] = n_missing
        
        return {
            'subject': subject_name,
            'file_path': str(csv_path),
            'total_samples': total_samples,
            'total_duration_sec': total_duration,
            'total_duration_min': total_duration / 60.0,
            'n_channels': n_channels,
            'channel_columns': channel_cols,
            'has_label': has_label,
            'has_phase': has_phase,
            'label_counts': label_counts,
            'channel_stats': channel_stats,
            'missing_by_phase': missing_by_phase,
            'columns': columns
        }
    
    except Exception as e:
        return {
            'subject': Path(csv_path).parent.name.replace('data_', ''),
            'file_path': str(csv_path),
            'error': str(e)
        }

def main():
    """Función principal de exploración."""
    print("="*80)
    print("PASO 1: EXPLORACIÓN INICIAL DE DATOS")
    print("="*80)
    print()
    
    # Buscar todos los archivos CSV
    csv_files = []
    data_path = Path(DATA_DIR)
    
    if not data_path.exists():
        print(f"ERROR: No se encontró el directorio {DATA_DIR}")
        print(f"Buscando en directorio raíz...")
        data_path = Path(BASE_DIR) / 'DATA' / 'Data-Experimento-Rafa'
    
    if data_path.exists():
        print(f"Buscando archivos CSV en: {data_path}")
        for data_dir in data_path.glob('data_*'):
            csv_files.extend(data_dir.glob('eeg_data_*.csv'))
    else:
        print(f"ERROR: No se encontró el directorio de datos")
        return
    
    if not csv_files:
        print("No se encontraron archivos CSV")
        return
    
    print(f"\nEncontrados {len(csv_files)} archivos CSV")
    print("-"*80)
    
    # Explorar cada archivo
    all_results = []
    for csv_file in sorted(csv_files):
        print(f"\nAnalizando: {csv_file.name}")
        result = explore_csv_file(csv_file)
        all_results.append(result)
        
        if 'error' in result:
            print(f"  ERROR: {result['error']}")
            continue
        
        print(f"  Sujeto: {result['subject']}")
        print(f"  Total muestras: {result['total_samples']:,}")
        print(f"  Duración total: {result['total_duration_min']:.2f} minutos ({result['total_duration_sec']:.1f} segundos)")
        print(f"  Canales: {result['n_channels']}")
        
        if result['label_counts']:
            print(f"  Fases encontradas:")
            for label, count in result['label_counts'].items():
                duration = count / 250.0
                print(f"    - {label}: {count:,} muestras ({duration:.1f} seg)")
    
    # Resumen general
    print("\n" + "="*80)
    print("RESUMEN GENERAL")
    print("="*80)
    
    # Tabla de sujetos
    print("\nTabla de sujetos:")
    print("-"*80)
    print(f"{'Sujeto':<15} {'Muestras':<12} {'Duración (min)':<15} {'Canales':<10} {'Fases':<10}")
    print("-"*80)
    
    for result in all_results:
        if 'error' in result:
            print(f"{result['subject']:<15} {'ERROR':<12} {'-':<15} {'-':<10} {'-':<10}")
            continue
        
        n_phases = len(result['label_counts']) if result['label_counts'] else 0
        print(f"{result['subject']:<15} {result['total_samples']:<12,} {result['total_duration_min']:<15.2f} {result['n_channels']:<10} {n_phases:<10}")
    
    # Análisis por fase
    print("\n" + "="*80)
    print("ANÁLISIS POR FASE")
    print("="*80)
    
    phase_summary = {}
    for label in LABELS_OF_INTEREST:
        phase_summary[label] = {
            'subjects': [],
            'total_samples': 0,
            'avg_samples': 0,
            'min_samples': float('inf'),
            'max_samples': 0
        }
    
    for result in all_results:
        if 'error' in result:
            continue
        
        for label in LABELS_OF_INTEREST:
            if label in result['label_counts']:
                count = result['label_counts'][label]
                phase_summary[label]['subjects'].append(result['subject'])
                phase_summary[label]['total_samples'] += count
                phase_summary[label]['min_samples'] = min(phase_summary[label]['min_samples'], count)
                phase_summary[label]['max_samples'] = max(phase_summary[label]['max_samples'], count)
    
    for label in LABELS_OF_INTEREST:
        summary = phase_summary[label]
        if summary['subjects']:
            summary['avg_samples'] = summary['total_samples'] / len(summary['subjects'])
            print(f"\n{label}:")
            print(f"  Sujetos con datos: {len(summary['subjects'])} ({', '.join(summary['subjects'])})")
            print(f"  Total muestras: {summary['total_samples']:,}")
            print(f"  Promedio: {summary['avg_samples']:.0f} muestras ({summary['avg_samples']/250:.1f} seg)")
            print(f"  Rango: {summary['min_samples']:,} - {summary['max_samples']:,} muestras")
        else:
            print(f"\n{label}: Sin datos")
    
    # Identificar problemas
    print("\n" + "="*80)
    print("IDENTIFICACIÓN DE PROBLEMAS")
    print("="*80)
    
    problems = []
    for result in all_results:
        if 'error' in result:
            problems.append(f"{result['subject']}: Error al leer archivo - {result['error']}")
            continue
        
        # Verificar fases faltantes
        missing_phases = [p for p in LABELS_OF_INTEREST if p not in result['label_counts']]
        if missing_phases:
            problems.append(f"{result['subject']}: Faltan fases - {', '.join(missing_phases)}")
        
        # Verificar datos muy cortos
        for label, count in result['label_counts'].items():
            if label in LABELS_OF_INTEREST:
                duration = count / 250.0
                if duration < 30:  # Menos de 30 segundos
                    problems.append(f"{result['subject']}: {label} muy corto ({duration:.1f} seg)")
        
        # Verificar canales con muchos NaN
        if result['channel_stats']:
            for ch_col, stats in result['channel_stats'].items():
                if stats['n_nan'] > result['total_samples'] * 0.1:  # Más del 10% NaN
                    problems.append(f"{result['subject']}: {ch_col} tiene {stats['n_nan']} valores NaN ({stats['n_nan']/result['total_samples']*100:.1f}%)")
    
    if problems:
        print("\nProblemas encontrados:")
        for i, problem in enumerate(problems, 1):
            print(f"  {i}. {problem}")
    else:
        print("\n✓ No se encontraron problemas evidentes")
    
    # Guardar resumen
    print("\n" + "="*80)
    print("EXPLORACIÓN COMPLETADA")
    print("="*80)
    print(f"\nTotal de sujetos analizados: {len([r for r in all_results if 'error' not in r])}")
    print(f"Total de archivos con errores: {len([r for r in all_results if 'error' in r])}")

if __name__ == '__main__':
    main()
