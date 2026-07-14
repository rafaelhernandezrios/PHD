"""
Código simple para conectar al Aura y graficar en tiempo real
Simple code to connect to Aura and plot in real-time
"""

from pylsl import StreamInlet, resolve_byprop
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import deque
import time

# Factor de escala para convertir datos raw a microvoltios
SCALE_FACTOR_EEG = (4500000) / 24 / (2**23 - 1)  # uV/count

# Parámetros de visualización
BUFFER_DURATION = 5.0  # segundos de datos a mostrar
UPDATE_INTERVAL_MS = 200  # actualizar gráfico cada 200ms
CHUNK_DURATION = 5.0  # procesar datos en chunks de 1 segundo
DECIMATION_FACTOR = 2  # Decimar: tomar 1 muestra cada N (reduce datos a graficar)

print("=" * 60)
print("Buscando stream AURA... / Looking for AURA stream...")
print("=" * 60)

# Buscar stream AURA
streams = resolve_byprop('name', 'AURA', timeout=5.0)

if len(streams) == 0:
    print("ERROR: No se encontró el stream AURA")
    print("ERROR: AURA stream not found")
    print("\nAsegúrate de que:")
    print("  - El dispositivo AURA esté conectado")
    print("  - El software AURA esté ejecutándose")
    print("\nMake sure:")
    print("  - AURA device is connected")
    print("  - AURA software is running")
    exit(1)

# Conectar al stream
inlet = StreamInlet(streams[0])

# Obtener información del stream
stream_info = inlet.info()
nominal_srate = stream_info.nominal_srate()
num_channels = stream_info.channel_count()

print(f"\n✓ Conectado a AURA / Connected to AURA")
print(f"  Frecuencia de muestreo / Sampling rate: {nominal_srate} Hz")
print(f"  Número de canales / Number of channels: {num_channels}")

# Calcular tamaño del buffer
BUFFER_SIZE = int(BUFFER_DURATION * nominal_srate)

# Crear buffers para cada canal
timestamps = deque(maxlen=BUFFER_SIZE)
data_buffers = [deque(maxlen=BUFFER_SIZE) for _ in range(num_channels)]

# Configurar gráfico
fig, axes = plt.subplots(num_channels, 1, figsize=(14, 2 * num_channels))
if num_channels == 1:
    axes = [axes]  # Asegurar que sea una lista

fig.suptitle('EEG en Tiempo Real - AURA / Real-time EEG - AURA', 
             fontsize=14, fontweight='bold')

# Configurar cada subplot
lines = []
for i, ax in enumerate(axes):
    ax.set_title(f'Canal {i+1} / Channel {i+1}', fontsize=10)
    ax.set_xlabel('Tiempo (s) / Time (s)', fontsize=9)
    ax.set_ylabel('Amplitud (μV)', fontsize=9)
    ax.grid(True, alpha=0.3)
    line, = ax.plot([], [], linewidth=1.0, color='blue', alpha=0.8)
    lines.append(line)
    ax.set_xlim(0, BUFFER_DURATION)
    ax.set_ylim(-200, 200)  # Rango inicial, se ajustará automáticamente

# Variables para control de tiempo
start_time = None
sample_count = [0]
last_update_time = [None]

def update_plot(frame):
    """Función de actualización para animación - procesa datos en chunks"""
    global start_time
    
    try:
        # Calcular cuántas muestras esperar en un chunk (1 segundo)
        chunk_size = int(nominal_srate * CHUNK_DURATION)
        
        # Obtener chunk de muestras (más eficiente que pull_sample)
        # timeout=0.1 para no bloquear mucho, pero obtener suficientes datos
        chunk, timestamps_chunk = inlet.pull_chunk(timeout=0.1, max_samples=chunk_size)
        
        if len(chunk) == 0:
            # Si no hay datos nuevos, solo actualizar gráfico
            pass
        else:
            # Convertir a numpy array
            chunk_array = np.array(chunk)  # Shape: (n_samples, n_channels)
            
            # Inicializar tiempo de inicio con el primer timestamp
            if start_time is None and len(timestamps_chunk) > 0:
                start_time = timestamps_chunk[0]
            
            # Procesar cada muestra del chunk
            for i, (sample, timestamp) in enumerate(zip(chunk_array, timestamps_chunk)):
                # Calcular tiempo relativo
                rel_time = timestamp - start_time
                
                # Escalar datos a microvoltios
                channel_data = sample * SCALE_FACTOR_EEG
                
                # Decimación: solo agregar cada DECIMATION_FACTOR muestras
                if i % DECIMATION_FACTOR == 0:
                    # Agregar a buffers (deque automáticamente elimina los más antiguos)
                    timestamps.append(rel_time)
                    for ch in range(num_channels):
                        if ch < len(channel_data):
                            data_buffers[ch].append(channel_data[ch])
                
                sample_count[0] += 1
    
    except Exception as e:
        # Ignorar errores menores
        pass
    
    # Actualizar gráficos si hay datos
    if len(timestamps) > 1:
        # Convertir a arrays numpy (más eficiente)
        time_array = np.array(timestamps)
        
        # Actualizar cada línea
        for ch, (line, data_buffer) in enumerate(zip(lines, data_buffers)):
            if len(data_buffer) > 0 and len(time_array) == len(data_buffer):
                data_array = np.array(data_buffer)
                
                # Actualizar datos de la línea
                line.set_data(time_array, data_array)
                
                # Ajustar límites de ejes X - ventana deslizante
                ax = line.axes
                if len(time_array) > 0:
                    current_time = time_array[-1]
                    if current_time > BUFFER_DURATION:
                        # Ventana deslizante: mostrar últimos BUFFER_DURATION segundos
                        # Esto hace que se "borre" lo de atrás automáticamente
                        ax.set_xlim(current_time - BUFFER_DURATION, current_time + 0.1)
                    else:
                        # Al inicio, mostrar desde 0 hasta BUFFER_DURATION
                        ax.set_xlim(0, BUFFER_DURATION)
                
                # Ajustar límites de eje Y usando percentiles (más robusto)
                if len(data_array) > 10:
                    y_min = np.percentile(data_array, 5)
                    y_max = np.percentile(data_array, 95)
                    y_range = y_max - y_min
                    if y_range > 0:
                        ax.set_ylim(y_min - 0.2 * y_range, y_max + 0.2 * y_range)
                    else:
                        ax.set_ylim(-100, 100)
        
        # Actualizar título con información
        if start_time is not None:
            current_rel_time = time_array[-1] if len(time_array) > 0 else 0
            effective_fs = nominal_srate / DECIMATION_FACTOR
            # Calcular retraso aproximado
            buffer_used = len(timestamps) / (nominal_srate / DECIMATION_FACTOR)
            fig.suptitle(
                f'EEG en Tiempo Real - AURA | Muestras: {sample_count[0]} | '
                f'Tiempo: {current_rel_time:.1f}s | Fs: {nominal_srate:.1f} Hz (graficando: {effective_fs:.1f} Hz) | '
                f'Buffer: {buffer_used:.1f}s | Canales: {num_channels}',
                fontsize=11, fontweight='bold')
    
    return lines

# Iniciar visualización
print("\n" + "=" * 60)
print("Iniciando visualización en tiempo real...")
print("Starting real-time visualization...")
print("=" * 60)
print(f"Buffer: {BUFFER_DURATION} segundos ({BUFFER_SIZE} muestras)")
print(f"Actualización cada {UPDATE_INTERVAL_MS}ms")
print(f"Procesamiento: chunks de {CHUNK_DURATION}s (más eficiente)")
print(f"Decimación: 1 muestra cada {DECIMATION_FACTOR} (Fs efectiva: {nominal_srate/DECIMATION_FACTOR:.1f} Hz)")
print(f"Canales a mostrar: {num_channels}")
print(f"\nNOTA: Los datos antiguos se eliminan automáticamente del buffer")
print(f"NOTE: Old data is automatically removed from buffer")
print("\nPresiona Ctrl+C para detener / Press Ctrl+C to stop")
print("-" * 60)

try:
    plt.tight_layout()
    ani = FuncAnimation(fig, update_plot, interval=UPDATE_INTERVAL_MS, 
                       blit=False, cache_frame_data=False)
    plt.show()
except KeyboardInterrupt:
    print("\n\nDeteniendo visualización... / Stopping visualization...")
    plt.close('all')
    print("Visualización detenida. / Visualization stopped.")

