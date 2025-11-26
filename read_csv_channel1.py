import pandas as pd
import numpy as np
import sys

# Nombre del archivo CSV (puedes cambiarlo o pasarlo como argumento)
if len(sys.argv) > 1:
    csv_file = sys.argv[1]
else:
    csv_file = "./aura_data_20251114_140333.csv"

print(f"Leyendo archivo: {csv_file}")

try:
    # Leer el CSV
    df = pd.read_csv(csv_file)
    
    # Verificar si existe la columna Channel_1 (probar diferentes variantes)
    channel_col = None
    for col_name in ['Channel_1', 'channel_1', 'CHANNEL_1', 'ch1']:
        if col_name in df.columns:
            channel_col = col_name
            break
    
    if channel_col is None:
        print("Error: No se encontró la columna Channel_1")
        print(f"Columnas disponibles: {list(df.columns)}")
        exit(1)
    
    # Obtener timestamps y datos de Channel_1
    timestamps = df['timestamp'].values
    channel_data = df[channel_col].values
    
    # Contar muestras (filas) - cada fila es una muestra
    num_samples = len(df)
    count_non_null = df[channel_col].notna().sum()
    
    # Calcular frecuencia de muestreo basada en timestamps
    if len(timestamps) > 1:
        # Calcular diferencias entre timestamps consecutivos
        time_diffs = np.diff(timestamps)
        avg_time_diff = np.mean(time_diffs)
        sampling_rate = 1.0 / avg_time_diff if avg_time_diff > 0 else 0
        
        # Calcular duración total
        total_duration = timestamps[-1] - timestamps[0]
        
        # Mostrar información
        print(f"\n{'='*60}")
        print(f"Análisis basado en un solo canal ({channel_col}):")
        print(f"{'='*60}")
        print(f"Número de muestras (filas): {num_samples}")
        print(f"Datos válidos en {channel_col}: {count_non_null}")
        print(f"\nCálculo de frecuencia de muestreo:")
        print(f"  Timestamp inicial: {timestamps[0]:.6f}")
        print(f"  Timestamp final: {timestamps[-1]:.6f}")
        print(f"  Duración total: {total_duration:.6f} segundos")
        print(f"  Diferencia promedio entre muestras: {avg_time_diff:.6f} segundos")
        print(f"  Frecuencia de muestreo: {sampling_rate:.2f} Hz")
        print(f"\nVerificación:")
        print(f"  Muestras esperadas (duración × frecuencia): {total_duration * sampling_rate:.0f}")
        print(f"  Muestras reales: {num_samples}")
        
        # Estadísticas de diferencias de tiempo
        print(f"\nEstadísticas de intervalos entre muestras:")
        print(f"  Mínimo: {np.min(time_diffs):.6f} s")
        print(f"  Máximo: {np.max(time_diffs):.6f} s")
        print(f"  Desviación estándar: {np.std(time_diffs):.6f} s")
    else:
        print("Error: No hay suficientes muestras para calcular la frecuencia")
    
except FileNotFoundError:
    print(f"Error: No se encontró el archivo {csv_file}")
except Exception as e:
    print(f"Error al leer el archivo: {e}")

