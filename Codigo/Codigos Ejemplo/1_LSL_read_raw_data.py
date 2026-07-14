from pylsl import StreamInlet, resolve_byprop, resolve_bypred
import matplotlib.pyplot as plt
import numpy as np
from collections import deque
import time
from scipy import signal

#from pylsl import StreamInfo, StreamOutlet, resolve_byprop

print("looking for an EEG stream...")
streams = resolve_byprop('name', 'AURA')
#streams = resolve_byprop('name', 'AURA_Filtered')
#streams = resolve_byprop('name', 'AURA_Power')

inlet = StreamInlet(streams[0])

# Asumir frecuencia de muestreo de AURA (250 Hz)
sampling_rate = 250.0  # Hz
print(f"Frecuencia de muestreo asumida / Assumed sampling rate: {sampling_rate} Hz")

# Parámetros del filtro bandpass
lowcut = 0.2  # Hz
highcut = 50.0  # Hz
nyquist = sampling_rate / 2.0
low = lowcut / nyquist
high = highcut / nyquist

# Diseñar el filtro Butterworth bandpass
b, a = signal.butter(4, [low, high], btype='band')
# Obtener condiciones iniciales para filtro con estado (tiempo real)
zi_bandpass = signal.lfilter_zi(b, a)
print(f"Filtro bandpass configurado: {lowcut}-{highcut} Hz / Bandpass filter configured: {lowcut}-{highcut} Hz")

# Parámetros del filtro stopband (notch) de 60 Hz
freq_notch = 60.0  # Hz - frecuencia a eliminar
bandwidth = 2.0  # Hz - ancho de banda del notch
low_stop = (freq_notch - bandwidth/2) / nyquist
high_stop = (freq_notch + bandwidth/2) / nyquist

# Crear el filtro stopband (bandstop)
b_stop, a_stop = signal.butter(4, [low_stop, high_stop], btype='bandstop')
# Obtener condiciones iniciales para filtro con estado (tiempo real)
zi_stopband = signal.lfilter_zi(b_stop, a_stop)
print(f"Filtro stopband configurado: {freq_notch} Hz (bandwidth: {bandwidth} Hz) / Stopband filter configured: {freq_notch} Hz (bandwidth: {bandwidth} Hz)")

# Configuración del gráfico en tiempo real
# Buffer para almacenar los últimos N valores (ventana de tiempo)
# 5 segundos a 250 Hz = 1250 muestras
BUFFER_SIZE = 1250  # Número de muestras a mostrar (5 segundos)
data_buffer = deque(maxlen=BUFFER_SIZE)  # Buffer para sample[1] (para visualización)
filtered_buffer = deque(maxlen=BUFFER_SIZE)  # Buffer para señal filtrada
car_buffer = deque(maxlen=BUFFER_SIZE)  # Buffer para señal con CAR
car_stopband_buffer = deque(maxlen=BUFFER_SIZE)  # Buffer para señal con CAR + stopband
all_channels_buffer = deque(maxlen=BUFFER_SIZE)  # Buffer para todos los 8 canales (para CAR)
all_channels_filtered_buffer = deque(maxlen=BUFFER_SIZE)  # Buffer para canales filtrados (para CAR)
time_buffer = deque(maxlen=BUFFER_SIZE)

# Estados de los filtros (para procesamiento en tiempo real)
# Estado del filtro bandpass para canal principal
zi_ch0 = zi_bandpass.copy()
# Estados del filtro bandpass para cada uno de los 8 canales (para CAR)
zi_channels = [zi_bandpass.copy() for _ in range(8)]
# Estado del filtro stopband
zi_stop = zi_stopband.copy()

# Configurar matplotlib en modo interactivo con cuatro subplots
plt.ion()
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 16), sharex=True)

# Gráfico 1: Señal original
line1, = ax1.plot([], [], 'b-', linewidth=1, label='Señal Original / Original Signal')
ax1.set_ylabel('EEG Amplitude (μV) - sample[1]', fontsize=12)
ax1.set_title('Gráfico en Tiempo Real - Canal EEG sample[1] (Original) / Real-time Plot - EEG Channel sample[1] (Original)', fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend()

# Gráfico 2: Señal filtrada
line2, = ax2.plot([], [], 'r-', linewidth=1, label=f'Señal Filtrada (Bandpass {lowcut}-{highcut} Hz) / Filtered Signal (Bandpass {lowcut}-{highcut} Hz)')
ax2.set_ylabel('EEG Amplitude (μV) - sample[1]', fontsize=12)
ax2.set_title(f'Gráfico en Tiempo Real - Canal EEG sample[1] (Filtrado Bandpass {lowcut}-{highcut} Hz) / Real-time Plot - EEG Channel sample[1] (Filtered Bandpass {lowcut}-{highcut} Hz)', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.legend()

# Gráfico 3: Señal filtrada + Common Average Reference (CAR)
line3, = ax3.plot([], [], 'g-', linewidth=1, label='Señal Filtrada + CAR / Filtered Signal + CAR')
ax3.set_ylabel('EEG Amplitude (μV) - sample[1]', fontsize=12)
ax3.set_title(f'Gráfico en Tiempo Real - Canal EEG sample[1] (Bandpass {lowcut}-{highcut} Hz + CAR) / Real-time Plot - EEG Channel sample[1] (Bandpass {lowcut}-{highcut} Hz + CAR)', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3)
ax3.legend()

# Gráfico 4: Señal filtrada + CAR + Stopband 60 Hz
line4, = ax4.plot([], [], 'm-', linewidth=1, label=f'Señal Filtrada + CAR + Stopband {freq_notch} Hz / Filtered Signal + CAR + Stopband {freq_notch} Hz')
ax4.set_xlabel('Tiempo (muestras) / Time (samples)', fontsize=12)
ax4.set_ylabel('EEG Amplitude (μV) - sample[1]', fontsize=12)
ax4.set_title(f'Gráfico en Tiempo Real - Canal EEG sample[1] (Bandpass {lowcut}-{highcut} Hz + CAR + Stopband {freq_notch} Hz) / Real-time Plot - EEG Channel sample[1] (Bandpass {lowcut}-{highcut} Hz + CAR + Stopband {freq_notch} Hz)', fontsize=12, fontweight='bold')
ax4.grid(True, alpha=0.3)
ax4.legend()

plt.tight_layout()

print("Iniciando gráfico en tiempo real... / Starting real-time plot...")
print("Presiona Ctrl+C para detener / Press Ctrl+C to stop")

try:
    while True:
        # get a new sample (you can also omit the timestamp part if you're not
        # interested in it)
        sample, timestamp = inlet.pull_sample()
        
        # Extraer sample[1] y todos los canales
        if len(sample) >= 2:
            sample_value = sample[1]  # Canal a graficar
            data_buffer.append(sample_value)
            all_channels_buffer.append(sample)  # Guardar todos los canales para CAR
            time_buffer.append(len(data_buffer) - 1)  # Usar índice como tiempo
            
            # PROCESAMIENTO EN TIEMPO REAL (solo la nueva muestra)
            # 1. Aplicar filtro bandpass al canal principal (sample[1])
            filtered_sample, zi_ch0 = signal.lfilter(b, a, [sample_value], zi=zi_ch0)
            filtered_value = filtered_sample[0]
            filtered_buffer.append(filtered_value)
            
            # 2. Aplicar filtro bandpass a todos los 8 canales (para CAR)
            filtered_channels_sample = np.zeros(8)
            for ch in range(min(8, len(sample))):
                filtered_ch, zi_channels[ch] = signal.lfilter(b, a, [sample[ch]], zi=zi_channels[ch])
                filtered_channels_sample[ch] = filtered_ch[0]
            all_channels_filtered_buffer.append(filtered_channels_sample)
            
            # 3. Calcular CAR: promedio de los 8 canales filtrados en esta muestra
            if len(all_channels_filtered_buffer) > 0:
                common_avg = np.mean(filtered_channels_sample)
                car_value = filtered_value - common_avg
            else:
                car_value = filtered_value
            car_buffer.append(car_value)
            
            # 4. Aplicar filtro stopband de 60 Hz después del CAR
            car_stopband_sample, zi_stop = signal.lfilter(b_stop, a_stop, [car_value], zi=zi_stop)
            car_stopband_value = car_stopband_sample[0]
            car_stopband_buffer.append(car_stopband_value)
            
            # Convertir buffers a arrays para visualización
            data_array = np.array(list(data_buffer))
            filtered_array = np.array(list(filtered_buffer))
            car_array = np.array(list(car_buffer))
            car_stopband_array = np.array(list(car_stopband_buffer))
            time_array = np.array(list(time_buffer))
            
            # Actualizar gráfico 1: señal original
            line1.set_data(time_array, data_array)
            ax1.relim()
            ax1.autoscale_view()
            
            # Actualizar gráfico 2: señal filtrada
            line2.set_data(time_array, filtered_array)
            ax2.relim()
            ax2.autoscale_view()
            
            # Actualizar gráfico 3: señal con CAR
            line3.set_data(time_array, car_array)
            ax3.relim()
            ax3.autoscale_view()
            
            # Actualizar gráfico 4: señal con CAR + Stopband 60 Hz
            line4.set_data(time_array, car_stopband_array)
            ax4.relim()
            ax4.autoscale_view()
            
            # Actualizar la ventana
            plt.draw()
            plt.pause(0.001)  # Pequeña pausa para permitir actualización
            
            # Opcional: imprimir cada muestra (puedes comentar esto si es muy rápido)
            # print(f"Timestamp: {timestamp:.4f}, sample[0]: {sample_value:.4f}")
            
except KeyboardInterrupt:
    print("\nDeteniendo gráfico... / Stopping plot...")
    plt.close('all')