from pylsl import StreamInlet, resolve_byprop
import numpy as np
import time
from scipy import signal
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import deque
import pywt

SCALE_FACTOR_EEG = (4500000)/24/(2**23-1)  # uV/count

# Parámetros de filtrado
LOWCUT = 0.2  # Hz
HIGHCUT = 50.0  # Hz
FREQ_NOTCH = 60.0  # Hz
BANDWIDTH_NOTCH = 2.0  # Hz

# Parámetros de filtrado por ventanas
WINDOW_DURATION = 4.0  # segundos
SAMPLING_RATE = 250.0  # Hz
WINDOW_SIZE = int(WINDOW_DURATION * SAMPLING_RATE)  # 1000 muestras a 250 Hz

# Parámetros de visualización
BUFFER_SIZE = 2000  # Número de muestras a mantener en el buffer (8 segundos)
UPDATE_INTERVAL = 10  # Actualizar gráfico cada N muestras procesadas
NUM_CHANNELS = 8

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

print(f"Conectado a AURA")
print(f"Samplerate nominal: {nominal_srate} Hz")
print(f"Número de canales: {num_channels}")

# Calcular parámetros de filtros
nyquist = nominal_srate / 2.0
low = LOWCUT / nyquist
high = HIGHCUT / nyquist
low_stop = (FREQ_NOTCH - BANDWIDTH_NOTCH/2) / nyquist
high_stop = (FREQ_NOTCH + BANDWIDTH_NOTCH/2) / nyquist

# Crear filtros
b_bandpass, a_bandpass = signal.butter(4, [low, high], btype='band')
b_notch, a_notch = signal.butter(4, [low_stop, high_stop], btype='bandstop')

# Buffers para ventanas de procesamiento (4 segundos)
window_raw = []  # Acumular muestras raw para procesar
window_timestamps = []  # Timestamps de la ventana

# Buffers de datos para visualización
timestamps = deque(maxlen=BUFFER_SIZE)
raw_data = [deque(maxlen=BUFFER_SIZE) for _ in range(num_channels)]
filtered_data = [deque(maxlen=BUFFER_SIZE) for _ in range(num_channels)]
car_data = [deque(maxlen=BUFFER_SIZE) for _ in range(num_channels)]
notch_data = [deque(maxlen=BUFFER_SIZE) for _ in range(num_channels)]

# Configurar gráficos
fig, axes = plt.subplots(4, 2, figsize=(16, 12))
fig.suptitle('Procesamiento EEG en Tiempo Real - AURA / Real-time EEG Processing - AURA', 
             fontsize=14, fontweight='bold')

# Configurar cada subplot
plot_configs = [
    ('Raw / Datos Raw', raw_data, 'red'),
    ('Bandpass Filtered / Filtrado Bandpass', filtered_data, 'blue'),
    ('CAR / Referencia Común', car_data, 'green'),
    ('Notch 60Hz / Notch 60Hz', notch_data, 'purple')
]

lines = []
for row, (title, data_buffer, color) in enumerate(plot_configs):
    for col in range(2):
        ax = axes[row, col]
        channel_idx = col  # Mostrar canales 0 y 1 (Channel 1 y 2)
        ax.set_title(f'{title} - Channel {channel_idx + 1}', fontsize=10)
        ax.set_xlabel('Tiempo (s) / Time (s)', fontsize=9)
        ax.set_ylabel('Amplitud (μV)', fontsize=9)
        ax.grid(True, alpha=0.3)
        line, = ax.plot([], [], linewidth=0.8, color=color, alpha=0.7)
        lines.append((line, data_buffer[channel_idx]))

# Función para aplicar filtros sobre una ventana completa
def apply_filters_window(window_data):
    """
    Aplica filtros sobre una ventana completa de datos usando filtfilt
    window_data: array de forma (n_samples, n_channels)
    """
    window_array = np.array(window_data)  # (n_samples, n_channels)
    
    # Transponer para tener (n_channels, n_samples)
    window_transposed = window_array.T
    
    # 1. Bandpass filter (aplicar a cada canal)
    filtered = np.zeros_like(window_transposed)
    for ch in range(num_channels):
        filtered[ch] = signal.filtfilt(b_bandpass, a_bandpass, window_transposed[ch])
    
    # 2. CAR (Common Average Reference) - calcular promedio por muestra
    car_mean = np.mean(filtered, axis=0)  # Promedio de todos los canales por muestra
    car_result = filtered - car_mean  # Restar promedio a cada canal
    
    # 3. Notch filter (aplicar a cada canal)
    notch_result = np.zeros_like(car_result)
    for ch in range(num_channels):
        notch_result[ch] = signal.filtfilt(b_notch, a_notch, car_result[ch])
    
    # Transponer de vuelta para tener (n_samples, n_channels)
    return filtered.T, car_result.T, notch_result.T

# Contador de muestras
sample_count = [0]
processed_count = [0]
start_time = [time.time()]
base_timestamp = [None]

def update_plot(frame):
    """Función de actualización para animación"""
    global window_raw, window_timestamps, base_timestamp
    
    try:
        # Obtener muestra (timeout muy corto para no bloquear mucho)
        sample, timestamp = inlet.pull_sample(timeout=0.001)
        
        if sample is None or len(sample) == 0:
            return lines
        
        # Inicializar timestamp base
        if base_timestamp[0] is None:
            base_timestamp[0] = timestamp
        
        # Escalar datos
        channel_data = np.array([float(x)*SCALE_FACTOR_EEG for x in sample])
        
        # Calcular tiempo relativo
        rel_time = timestamp - base_timestamp[0]
        
        # Agregar a ventana de procesamiento
        window_raw.append(channel_data)
        window_timestamps.append(rel_time)
        
        # Agregar raw a buffer de visualización inmediatamente
        timestamps.append(rel_time)
        for ch in range(num_channels):
            raw_data[ch].append(channel_data[ch])
        
        sample_count[0] += 1
        
        # Cuando la ventana esté llena (4 segundos = 1000 muestras a 250 Hz)
        if len(window_raw) >= WINDOW_SIZE:
            # Procesar ventana completa
            filtered_window, car_window, notch_window = apply_filters_window(window_raw)
            
            # Agregar resultados procesados a buffers de visualización
            # Usar los timestamps de la ventana para sincronización
            for i in range(len(window_raw)):
                for ch in range(num_channels):
                    filtered_data[ch].append(filtered_window[i, ch])
                    car_data[ch].append(car_window[i, ch])
                    notch_data[ch].append(notch_window[i, ch])
            
            processed_count[0] += len(window_raw)
            
            # Limpiar ventana (mantener las últimas muestras para solapamiento si se desea)
            # Por ahora, limpiar completamente
            window_raw = []
            window_timestamps = []
        
        # Actualizar gráficos cada UPDATE_INTERVAL muestras procesadas
        if processed_count[0] > 0 and processed_count[0] % UPDATE_INTERVAL == 0:
            if len(timestamps) > 1:
                time_array = np.array(timestamps)
                
                # Actualizar cada línea
                for idx, (line, data_buffer) in enumerate(lines):
                    if len(data_buffer) > 0:
                        line.set_data(time_array, np.array(data_buffer))
                        
                        # Ajustar límites de ejes
                        ax = line.axes
                        ax.relim()
                        ax.autoscale_view()
                
                # Actualizar título con información
                elapsed = time.time() - start_time[0]
                window_info = f"Ventana: {WINDOW_DURATION}s ({WINDOW_SIZE} muestras)"
                fig.suptitle(
                    f'Procesamiento EEG en Tiempo Real - AURA | Muestras: {sample_count[0]} | Procesadas: {processed_count[0]} | Tiempo: {elapsed:.1f}s | Fs: {nominal_srate:.1f} Hz | {window_info}',
                    fontsize=11, fontweight='bold')
        
    except Exception as e:
        print(f"Error en actualización: {e}")
        import traceback
        traceback.print_exc()
    
    return lines

# Iniciar animación
print("\n" + "="*60)
print("Iniciando visualización en tiempo real...")
print("="*60)
print("Filtros aplicados:")
print(f"  - Bandpass: {LOWCUT}-{HIGHCUT} Hz")
print(f"  - CAR (Common Average Reference)")
print(f"  - Notch: {FREQ_NOTCH} Hz (±{BANDWIDTH_NOTCH/2} Hz)")
print(f"\nProcesamiento por ventanas:")
print(f"  - Duración de ventana: {WINDOW_DURATION} segundos")
print(f"  - Tamaño de ventana: {WINDOW_SIZE} muestras (a {SAMPLING_RATE} Hz)")
print(f"  - Método: filtfilt sobre ventanas completas")
print(f"\nVisualización:")
print(f"  - Canales mostrados: 1 y 2 de {num_channels} canales")
print(f"  - Buffer de visualización: {BUFFER_SIZE} muestras ({BUFFER_SIZE/SAMPLING_RATE:.1f} segundos)")
print(f"  - Actualización cada {UPDATE_INTERVAL} muestras procesadas")
print("\nPresiona Ctrl+C para detener")
print("-" * 60)

try:
    ani = FuncAnimation(fig, update_plot, interval=50, blit=False, cache_frame_data=False)
    plt.tight_layout()
    plt.show()
except KeyboardInterrupt:
    print("\n\nDeteniendo visualización...")
    plt.close('all')
    print("Visualización detenida.")

