"""
signal_worker.py
Clase QThread para adquisición de datos EEG vía LSL y procesamiento de señal en tiempo real.
Maneja filtrado, buffer circular y cálculo de bandpower.
"""

import numpy as np
from scipy import signal
from PyQt5.QtCore import QThread, pyqtSignal
from pylsl import StreamInlet, resolve_byprop
from collections import deque
import time


class RingBuffer:
    """
    Buffer circular para almacenar datos de señal EEG.
    Permite cálculo de FFT en ventanas móviles.
    """
    def __init__(self, maxlen, n_channels=8):
        """
        Args:
            maxlen: Tamaño máximo del buffer (número de muestras)
            n_channels: Número de canales EEG
        """
        self.maxlen = maxlen
        self.n_channels = n_channels
        self.buffer = np.zeros((maxlen, n_channels))
        self.timestamps = deque(maxlen=maxlen)
        self.write_idx = 0
        self.is_full = False
        
    def append(self, data, timestamp):
        """
        Añade una nueva muestra al buffer.
        
        Args:
            data: Array de shape (n_channels,) con los valores de los canales
            timestamp: Timestamp de la muestra
        """
        self.buffer[self.write_idx] = data
        self.timestamps.append(timestamp)
        self.write_idx = (self.write_idx + 1) % self.maxlen
        if self.write_idx == 0:
            self.is_full = True
    
    def get_window(self, window_samples):
        """
        Obtiene la última ventana de muestras.
        
        Args:
            window_samples: Número de muestras a retornar
            
        Returns:
            Array de shape (window_samples, n_channels) con los datos más recientes
        """
        if not self.is_full and self.write_idx < window_samples:
            return self.buffer[:self.write_idx]
        
        start_idx = (self.write_idx - window_samples) % self.maxlen
        if start_idx + window_samples <= self.maxlen:
            return self.buffer[start_idx:start_idx + window_samples]
        else:
            # Caso donde la ventana cruza el límite del buffer
            part1 = self.buffer[start_idx:]
            part2 = self.buffer[:start_idx + window_samples - self.maxlen]
            return np.vstack([part1, part2])


class SignalWorker(QThread):
    """
    Worker thread para adquisición y procesamiento de señal EEG.
    No bloquea la interfaz gráfica durante la adquisición.
    """
    
    # Señales PyQt para comunicación con la UI
    data_ready = pyqtSignal(np.ndarray, float)  # datos procesados, timestamp
    raw_data_ready = pyqtSignal(np.ndarray, float)  # datos sin filtrar, timestamp
    connection_status = pyqtSignal(bool, str)  # conectado, mensaje
    
    def __init__(self, sample_rate=250, n_channels=8, buffer_duration=2.0):
        """
        Args:
            sample_rate: Tasa de muestreo en Hz (250 Hz para AURA)
            n_channels: Número de canales EEG (8 para AURA)
            buffer_duration: Duración del buffer en segundos (2.0 s para ventana FFT)
        """
        super().__init__()
        self.sample_rate = sample_rate
        self.n_channels = n_channels
        self.buffer_samples = int(buffer_duration * sample_rate)
        
        # Buffer circular
        self.ring_buffer = RingBuffer(maxlen=self.buffer_samples * 2, n_channels=n_channels)
        
        # Filtros
        self._setup_filters()
        
        # Control del thread
        self.running = False
        self.inlet = None
        
        # Índices de canales para análisis (Fz = canal 0, Pz = canal 4)
        self.fz_channel = 0
        self.pz_channel = 4
        
        # Buffer para acumular muestras antes de emitir (reduce carga en UI)
        self.plot_buffer = []
        self.plot_buffer_size = 20  # Emitir cada 20 muestras (~80ms a 250Hz, aumentado para mejor rendimiento)
        self.last_plot_time = 0
        self.plot_interval = 0.1  # Emitir cada 100ms máximo (aumentado de 40ms)
    
    def _setup_filters(self):
        """Configura los filtros digitales para procesamiento de señal."""
        # Filtro pasabanda 1-40 Hz (Butterworth, orden 4)
        nyquist = self.sample_rate / 2
        low = 1.0 / nyquist
        high = 40.0 / nyquist
        self.b, self.a = signal.butter(4, [low, high], btype='band')
        
        # Filtro Notch 60 Hz (para eliminar ruido de línea eléctrica)
        notch_freq = 60.0
        quality_factor = 30.0
        self.b_notch, self.a_notch = signal.iirnotch(notch_freq, quality_factor, self.sample_rate)
        
        # Estado inicial de los filtros (uno por canal)
        zi_band_single = signal.lfilter_zi(self.b, self.a)
        zi_notch_single = signal.lfilter_zi(self.b_notch, self.a_notch)
        self.zi_band = np.tile(zi_band_single[:, np.newaxis], (1, self.n_channels))
        self.zi_notch = np.tile(zi_notch_single[:, np.newaxis], (1, self.n_channels))
    
    def connect_to_stream(self):
        """Busca y conecta al stream LSL de AURA."""
        try:
            print("Buscando stream EEG AURA...")
            streams = resolve_byprop('name', 'AURA', timeout=1.0)
            
            if len(streams) == 0:
                self.connection_status.emit(False, "No se encontró el stream AURA")
                return False
            
            self.inlet = StreamInlet(streams[0])
            info = self.inlet.info()
            print(f"Conectado a: {info.name()}")
            print(f"Canales: {info.channel_count()}")
            print(f"Sample rate: {info.nominal_srate()}")
            
            self.connection_status.emit(True, f"Conectado a {info.name()}")
            return True
            
        except Exception as e:
            error_msg = f"Error al conectar: {str(e)}"
            print(error_msg)
            self.connection_status.emit(False, error_msg)
            return False
    
    def calculate_bandpower(self, signal_data, freq_band, sample_rate):
        """
        Calcula la potencia espectral en una banda de frecuencias usando Welch's method.
        
        Args:
            signal_data: Array 1D con los datos de señal
            freq_band: Tupla (fmin, fmax) con los límites de la banda
            sample_rate: Tasa de muestreo
            
        Returns:
            Potencia promedio en la banda especificada
        """
        if len(signal_data) < sample_rate // 2:  # Necesitamos al menos 0.5 segundos de datos
            return 0.0
        
        # Método de Welch para estimación espectral
        nperseg = min(len(signal_data), sample_rate)
        freqs, psd = signal.welch(signal_data, sample_rate, nperseg=nperseg, noverlap=nperseg//2)
        
        # Encontrar índices de la banda de frecuencia
        idx_band = np.logical_and(freqs >= freq_band[0], freqs <= freq_band[1])
        
        # Calcular potencia promedio en la banda
        bandpower = np.trapz(psd[idx_band], freqs[idx_band])
        
        return bandpower
    
    def get_cognitive_load_ratio(self):
        """
        Calcula el ratio de carga cognitiva: Theta_Fz / Alpha_Pz
        
        Returns:
            Ratio de carga cognitiva o None si no hay suficientes datos
        """
        window_data = self.ring_buffer.get_window(self.buffer_samples)
        
        if len(window_data) < self.buffer_samples:
            return None
        
        # Extraer canales Fz y Pz
        fz_signal = window_data[:, self.fz_channel]
        pz_signal = window_data[:, self.pz_channel]
        
        # Calcular bandpower
        theta_band = (4.0, 7.0)  # Theta: 4-7 Hz
        alpha_band = (8.0, 12.0)  # Alpha: 8-12 Hz
        
        theta_power = self.calculate_bandpower(fz_signal, theta_band, self.sample_rate)
        alpha_power = self.calculate_bandpower(pz_signal, alpha_band, self.sample_rate)
        
        if alpha_power > 0:
            ratio = theta_power / alpha_power
            return ratio, theta_power, alpha_power
        
        return None
    
    def run(self):
        """Loop principal del thread. Adquiere y procesa datos continuamente."""
        if not self.inlet:
            if not self.connect_to_stream():
                return
        
        self.running = True
        print("Iniciando adquisición de datos...")
        
        while self.running:
            try:
                # Pull sample from LSL (timeout de 0.1 segundos)
                sample, timestamp = self.inlet.pull_sample(timeout=0.1)
                
                if sample:
                    # Convertir a numpy array
                    sample_array = np.array(sample[:self.n_channels])
                    
                    # DEBUG: Imprimir datos recibidos (solo las primeras 10 muestras)
                    if not hasattr(self, '_debug_counter'):
                        self._debug_counter = 0
                    if self._debug_counter < 10:
                        print(f"\n[Muestra {self._debug_counter}]")
                        print(f"  Tipo de sample: {type(sample)}")
                        print(f"  Longitud de sample: {len(sample)}")
                        print(f"  Sample completo: {sample}")
                        print(f"  sample_array shape: {sample_array.shape}")
                        print(f"  sample_array valores: {sample_array}")
                        print(f"  Timestamp: {timestamp}")
                        self._debug_counter += 1
                    
                    # Aplicar filtros (procesar cada muestra individualmente)
                    # Notch filter
                    filtered_notch = np.zeros(self.n_channels)
                    for i in range(self.n_channels):
                        filtered_notch[i], self.zi_notch[:, i] = signal.lfilter(
                            self.b_notch, self.a_notch, [sample_array[i]],
                            zi=self.zi_notch[:, i]
                        )
                    
                    # Bandpass filter
                    filtered_sample = np.zeros(self.n_channels)
                    for i in range(self.n_channels):
                        filtered_sample[i], self.zi_band[:, i] = signal.lfilter(
                            self.b, self.a, [filtered_notch[i]],
                            zi=self.zi_band[:, i]
                        )
                    
                    # Añadir al buffer
                    self.ring_buffer.append(filtered_sample, timestamp)
                    
                    # Acumular muestras para emitir en lotes (reduce saturación)
                    current_time = time.time()
                    self.plot_buffer.append((sample_array, filtered_sample, timestamp))
                    
                    # Emitir en lotes o cuando pase el intervalo máximo
                    if (len(self.plot_buffer) >= self.plot_buffer_size or 
                        (current_time - self.last_plot_time) >= self.plot_interval):
                        if self.plot_buffer:
                            # Emitir la última muestra del buffer
                            last_raw, last_filtered, last_ts = self.plot_buffer[-1]
                            self.raw_data_ready.emit(last_raw, last_ts)
                            self.data_ready.emit(last_filtered, last_ts)
                            self.plot_buffer.clear()
                            self.last_plot_time = current_time
                
            except Exception as e:
                print(f"Error en adquisición: {str(e)}")
                time.sleep(0.01)  # Pequeña pausa para evitar loops infinitos
    
    def stop(self):
        """Detiene la adquisición de datos."""
        self.running = False
        print("Deteniendo adquisición de datos...")

