from pylsl import StreamInlet, resolve_byprop, resolve_bypred
import numpy as np
import time
import csv
from datetime import datetime

SCALE_FACTOR_EEG = (4500000)/24/(2**23-1) #uV/count

print("Buscando stream AURA...")
streams = resolve_byprop('name', 'AURA')

if len(streams) == 0:
    print("No se encontró el stream AURA")
    exit(1)

inlet = StreamInlet(streams[0])

# Obtener información del stream
stream_info = inlet.info()
nominal_srate = stream_info.nominal_srate()
num_channels = stream_info.channel_count()

# Obtener nombres de canales (usar nombres genéricos si no se pueden obtener)
channel_names = []
try:
    channels_desc = stream_info.desc().child("channels")
    if not channels_desc.empty():
        ch = channels_desc.child("channel")
        for i in range(num_channels):
            if not ch.empty():
                ch_name = ch.child_value("label")
                if ch_name and ch_name != "":
                    channel_names.append(ch_name)
                else:
                    channel_names.append(f"Channel_{i+1}")
            else:
                channel_names.append(f"Channel_{i+1}")
            ch = ch.next_sibling()
    else:
        channel_names = [f"Channel_{i+1}" for i in range(num_channels)]
except:
    # Si hay algún error, usar nombres genéricos
    channel_names = [f"Channel_{i+1}" for i in range(num_channels)]

print(f"Conectado a AURA")
print(f"Samplerate nominal: {nominal_srate} Hz")
print(f"Número de canales: {num_channels}")
print(f"Nombres de canales: {channel_names}")

# Preparar datos para CSV
data_rows = []
recording_duration = 5.0  # 5 segundos

print(f"\nIniciando grabación de {recording_duration} segundos...")
start_time = time.time()

while True:
    sample, timestamp = inlet.pull_sample()
    channel_data = [float(x)*SCALE_FACTOR_EEG for x in sample]
    
    elapsed_time = time.time() - start_time
    
    # Guardar timestamp y datos de canales
    row = [timestamp] + channel_data
    data_rows.append(row)
    
    # Mostrar progreso
    if len(data_rows) % 50 == 0:
        print(f"Tiempo: {elapsed_time:.2f}s | Muestras: {len(data_rows)}")
    
    # Detener después de 5 segundos
    if elapsed_time >= recording_duration:
        break

print(f"\nGrabación completada: {len(data_rows)} muestras en {elapsed_time:.2f} segundos")

# Guardar en CSV
timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"aura_data_{timestamp_str}.csv"

with open(filename, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    
    # Escribir encabezados
    headers = ['timestamp'] + channel_names
    writer.writerow(headers)
    
    # Escribir datos
    writer.writerows(data_rows)

print(f"Datos guardados en: {filename}")
print(f"Total de muestras: {len(data_rows)}")

# Calcular frecuencia de muestreo basada en timestamps de un solo canal
print(f"\n{'='*60}")
print("Calculando frecuencia de muestreo basada en timestamps...")
print(f"{'='*60}")

if len(data_rows) > 1:
    # Extraer timestamps (primer elemento de cada fila)
    timestamps = [row[0] for row in data_rows]
    
    # Calcular diferencias entre timestamps consecutivos
    time_diffs = np.diff(timestamps)
    avg_time_diff = np.mean(time_diffs)
    sampling_rate = 1.0 / avg_time_diff if avg_time_diff > 0 else 0
    
    # Calcular duración total
    total_duration = timestamps[-1] - timestamps[0]
    
    print(f"Timestamp inicial: {timestamps[0]:.6f}")
    print(f"Timestamp final: {timestamps[-1]:.6f}")
    print(f"Duración total: {total_duration:.6f} segundos")
    print(f"Diferencia promedio entre muestras: {avg_time_diff:.6f} segundos")
    print(f"Frecuencia de muestreo calculada: {sampling_rate:.2f} Hz")
    print(f"\nVerificación:")
    print(f"  Muestras esperadas (duración × frecuencia): {total_duration * sampling_rate:.0f}")
    print(f"  Muestras reales: {len(data_rows)}")
    print(f"  Diferencia: {abs(len(data_rows) - total_duration * sampling_rate):.0f} muestras")
    
    # Estadísticas de diferencias de tiempo
    print(f"\nEstadísticas de intervalos entre muestras:")
    print(f"  Mínimo: {np.min(time_diffs):.6f} s")
    print(f"  Máximo: {np.max(time_diffs):.6f} s")
    print(f"  Desviación estándar: {np.std(time_diffs):.6f} s")
else:
    print("Error: No hay suficientes muestras para calcular la frecuencia")
    