"""
signal_worker.py
QThread class for EEG data acquisition via LSL and real-time signal processing.
Handles filtering, circular buffer and bandpower calculation.
"""

import numpy as np
from scipy import signal
from PyQt5.QtCore import QThread, pyqtSignal
from pylsl import StreamInlet, resolve_byprop
from collections import deque
import time


class RingBuffer:
    """
    Circular buffer for storing EEG signal data.
    Allows FFT calculation in moving windows.
    """
    def __init__(self, maxlen, n_channels=8):
        """
        Args:
            maxlen: Maximum buffer size (number of samples)
            n_channels: Number of EEG channels
        """
        self.maxlen = maxlen
        self.n_channels = n_channels
        self.buffer = np.zeros((maxlen, n_channels))
        self.timestamps = deque(maxlen=maxlen)
        self.write_idx = 0
        self.is_full = False
        
    def append(self, data, timestamp):
        """
        Adds a new sample to the buffer.
        
        Args:
            data: Array of shape (n_channels,) with channel values
            timestamp: Sample timestamp
        """
        self.buffer[self.write_idx] = data
        self.timestamps.append(timestamp)
        self.write_idx = (self.write_idx + 1) % self.maxlen
        if self.write_idx == 0:
            self.is_full = True
    
    def get_window(self, window_samples):
        """
        Gets the latest window of samples.
        
        Args:
            window_samples: Number of samples to return
            
        Returns:
            Array of shape (window_samples, n_channels) with the most recent data
        """
        if not self.is_full and self.write_idx < window_samples:
            return self.buffer[:self.write_idx]
        
        start_idx = (self.write_idx - window_samples) % self.maxlen
        if start_idx + window_samples <= self.maxlen:
            return self.buffer[start_idx:start_idx + window_samples]
        else:
            # Case where window crosses buffer boundary
            part1 = self.buffer[start_idx:]
            part2 = self.buffer[:start_idx + window_samples - self.maxlen]
            return np.vstack([part1, part2])


class SignalWorker(QThread):
    """
    Worker thread for EEG signal acquisition and processing.
    Does not block the graphical interface during acquisition.
    """
    
    # PyQt signals for communication with UI
    data_ready = pyqtSignal(np.ndarray, float)  # processed data, timestamp
    raw_data_ready = pyqtSignal(np.ndarray, float)  # unfiltered data, timestamp
    connection_status = pyqtSignal(bool, str)  # connected, message
    
    def __init__(self, sample_rate=250, n_channels=8, buffer_duration=2.0):
        """
        Args:
            sample_rate: Sampling rate in Hz (250 Hz for AURA)
            n_channels: Number of EEG channels (8 for AURA)
            buffer_duration: Buffer duration in seconds (2.0 s for FFT window)
        """
        super().__init__()
        self.sample_rate = sample_rate
        self.n_channels = n_channels
        self.buffer_samples = int(buffer_duration * sample_rate)
        
        # Circular buffer
        self.ring_buffer = RingBuffer(maxlen=self.buffer_samples * 2, n_channels=n_channels)
        
        # Filters
        self._setup_filters()
        
        # Thread control
        self.running = False
        self.inlet = None
        
        # Channel indices for analysis (Fz = channel 3, Pz = channel 6)
        # Channel mapping: 0=Fp1, 1=Fp2, 2=F3, 3=Fz, 4=F4, 5=P3, 6=Pz, 7=P4
        self.fz_channel = 3
        self.pz_channel = 6
        
        # Buffer to accumulate samples before emitting (reduces UI load)
        self.plot_buffer = []
        self.plot_buffer_size = 20  # Emit every 20 samples (~80ms at 250Hz, increased for better performance)
        self.last_plot_time = 0
        self.plot_interval = 0.1  # Emit every 100ms maximum (increased from 40ms)
    
    def _setup_filters(self):
        """Configures digital filters for signal processing."""
        # Bandpass filter 1-40 Hz (Butterworth, order 4)
        nyquist = self.sample_rate / 2
        low = 1.0 / nyquist
        high = 40.0 / nyquist
        self.b, self.a = signal.butter(4, [low, high], btype='band')
        
        # Notch filter 60 Hz (to eliminate electrical line noise)
        notch_freq = 60.0
        quality_factor = 30.0
        self.b_notch, self.a_notch = signal.iirnotch(notch_freq, quality_factor, self.sample_rate)
        
        # Initial filter state (one per channel)
        zi_band_single = signal.lfilter_zi(self.b, self.a)
        zi_notch_single = signal.lfilter_zi(self.b_notch, self.a_notch)
        self.zi_band = np.tile(zi_band_single[:, np.newaxis], (1, self.n_channels))
        self.zi_notch = np.tile(zi_notch_single[:, np.newaxis], (1, self.n_channels))
    
    def connect_to_stream(self):
        """Searches for and connects to AURA LSL stream."""
        try:
            print("Searching for AURA EEG stream...")
            streams = resolve_byprop('name', 'AURA', timeout=1.0)
            
            if len(streams) == 0:
                self.connection_status.emit(False, "AURA stream not found")
                return False
            
            self.inlet = StreamInlet(streams[0])
            info = self.inlet.info()
            print(f"Connected to: {info.name()}")
            print(f"Channels: {info.channel_count()}")
            print(f"Sample rate: {info.nominal_srate()}")
            
            self.connection_status.emit(True, f"Connected to {info.name()}")
            return True
            
        except Exception as e:
            error_msg = f"Connection error: {str(e)}"
            print(error_msg)
            self.connection_status.emit(False, error_msg)
            return False
    
    def calculate_bandpower(self, signal_data, freq_band, sample_rate):
        """
        Calculates spectral power in a frequency band using Welch's method.
        
        Args:
            signal_data: 1D array with signal data
            freq_band: Tuple (fmin, fmax) with band limits
            sample_rate: Sampling rate
            
        Returns:
            Average power in the specified band
        """
        if len(signal_data) < sample_rate // 2:  # We need at least 0.5 seconds of data
            return 0.0
        
        # Welch's method for spectral estimation
        nperseg = min(len(signal_data), sample_rate)
        freqs, psd = signal.welch(signal_data, sample_rate, nperseg=nperseg, noverlap=nperseg//2)
        
        # Find frequency band indices
        idx_band = np.logical_and(freqs >= freq_band[0], freqs <= freq_band[1])
        
        # Calculate average power in the band
        bandpower = np.trapz(psd[idx_band], freqs[idx_band])
        
        return bandpower
    
    def get_cognitive_load_ratio(self):
        """
        Calculates the cognitive load ratio: Theta_Fz / Alpha_Pz
        
        Returns:
            Cognitive load ratio or None if there's not enough data
        """
        window_data = self.ring_buffer.get_window(self.buffer_samples)
        
        if len(window_data) < self.buffer_samples:
            return None
        
        # Extract Fz and Pz channels
        fz_signal = window_data[:, self.fz_channel]
        pz_signal = window_data[:, self.pz_channel]
        
        # Calculate bandpower
        theta_band = (4.0, 7.0)  # Theta: 4-7 Hz
        alpha_band = (8.0, 12.0)  # Alpha: 8-12 Hz
        
        theta_power = self.calculate_bandpower(fz_signal, theta_band, self.sample_rate)
        alpha_power = self.calculate_bandpower(pz_signal, alpha_band, self.sample_rate)
        
        if alpha_power > 0:
            ratio = theta_power / alpha_power
            return ratio, theta_power, alpha_power
        
        return None
    
    def run(self):
        """Main thread loop. Acquires and processes data continuously."""
        if not self.inlet:
            if not self.connect_to_stream():
                return
        
        self.running = True
        print("Starting data acquisition...")
        
        while self.running:
            try:
                # Pull sample from LSL (0.1 second timeout)
                sample, timestamp = self.inlet.pull_sample(timeout=0.1)
                
                if sample:
                    # Convert to numpy array
                    sample_array = np.array(sample[:self.n_channels])
                    
                    # DEBUG: Print received data (only first 10 samples)
                    if not hasattr(self, '_debug_counter'):
                        self._debug_counter = 0
                    if self._debug_counter < 10:
                        print(f"\n[Sample {self._debug_counter}]")
                        print(f"  Sample type: {type(sample)}")
                        print(f"  Sample length: {len(sample)}")
                        print(f"  Full sample: {sample}")
                        print(f"  sample_array shape: {sample_array.shape}")
                        print(f"  sample_array values: {sample_array}")
                        print(f"  Timestamp: {timestamp}")
                        self._debug_counter += 1
                    
                    # Apply filters (process each sample individually)
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
                    
                    # Add to buffer
                    self.ring_buffer.append(filtered_sample, timestamp)
                    
                    # Accumulate samples to emit in batches (reduces saturation)
                    current_time = time.time()
                    self.plot_buffer.append((sample_array, filtered_sample, timestamp))
                    
                    # Emit in batches or when maximum interval passes
                    if (len(self.plot_buffer) >= self.plot_buffer_size or 
                        (current_time - self.last_plot_time) >= self.plot_interval):
                        if self.plot_buffer:
                            # Emit the last sample from buffer
                            last_raw, last_filtered, last_ts = self.plot_buffer[-1]
                            self.raw_data_ready.emit(last_raw, last_ts)
                            self.data_ready.emit(last_filtered, last_ts)
                            self.plot_buffer.clear()
                            self.last_plot_time = current_time
                
            except Exception as e:
                print(f"Acquisition error: {str(e)}")
                time.sleep(0.01)  # Small pause to avoid infinite loops
    
    def stop(self):
        """Stops data acquisition."""
        self.running = False
        print("Stopping data acquisition...")

