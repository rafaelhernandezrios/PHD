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
UPDATE_INTERVAL_MS = 50  # actualizar gráfico cada 50ms

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

def update_plot(frame):
    """Función de actualización para animación"""
    global start_time
    
    try:
        # Obtener muestra (timeout corto para no bloquear)
        sample, timestamp = inlet.pull_sample(timeout=0.001)
        
        if sample is None or len(sample) == 0:
            return lines
        
        # Inicializar tiempo de inicio
        if start_time is None:
            start_time = timestamp
        
        # Calcular tiempo relativo
        rel_time = timestamp - start_time
        
        # Escalar datos a microvoltios
        channel_data = np.array([float(x) * SCALE_FACTOR_EEG for x in sample])
        
        # Agregar a buffers
        timestamps.append(rel_time)
        for ch in range(num_channels):
            if ch < len(channel_data):
                data_buffers[ch].append(channel_data[ch])
        
        sample_count[0] += 1
        
        # Actualizar gráficos si hay datos suficientes
        if len(timestamps) > 1:
            time_array = np.array(timestamps)
            
            # Actualizar cada línea
            for ch, (line, data_buffer) in enumerate(zip(lines, data_buffers)):
                if len(data_buffer) > 0:
                    data_array = np.array(data_buffer)
                    line.set_data(time_array, data_array)
                    
                    # Ajustar límites de ejes
                    ax = line.axes
                    if len(time_array) > 0:
                        ax.set_xlim(max(0, time_array[-1] - BUFFER_DURATION), 
                                   max(BUFFER_DURATION, time_array[-1] + 0.5))
                    
                    if len(data_array) > 0:
                        y_min, y_max = np.min(data_array), np.max(data_array)
                        y_range = y_max - y_min
                        if y_range > 0:
                            ax.set_ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)
            
            # Actualizar título con información
            elapsed = time.time() - start_time if start_time else 0
            fig.suptitle(
                f'EEG en Tiempo Real - AURA | Muestras: {sample_count[0]} | '
                f'Tiempo: {elapsed:.1f}s | Fs: {nominal_srate:.1f} Hz | '
                f'Canales: {num_channels}',
                fontsize=11, fontweight='bold')
    
    except Exception as e:
        print(f"Error en actualización / Update error: {e}")
        import traceback
        traceback.print_exc()
    
    return lines

# Iniciar visualización
print("\n" + "=" * 60)
print("Iniciando visualización en tiempo real...")
print("Starting real-time visualization...")
print("=" * 60)
print(f"Buffer: {BUFFER_DURATION} segundos ({BUFFER_SIZE} muestras)")
print(f"Actualización cada {UPDATE_INTERVAL_MS}ms")
print(f"Canales a mostrar: {num_channels}")
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

