"""
signal_worker.py
QThread class for EEG data acquisition via LSL and real-time signal processing.
Handles filtering, circular buffer and bandpower calculation.
"""

import numpy as np
from scipy import signal
from PyQt5.QtCore import QThread, pyqtSignal
from pylsl import StreamInlet, resolve_byprop, resolve_streams
from collections import deque
import time
import json
from datetime import datetime

DEBUG_LOG_PATH = "/Users/rafael/Documents/Doctorado/PHD/.cursor/debug-805260.log"
DEBUG_SESSION_ID = "805260"

# #region debug_mode_logging (per-session NDJSON for this debug run)
DEBUG_MODE_LOG_PATH = "/Users/rafael/Documents/Doctorado/PHD/.cursor/debug-f8feef.log"
DEBUG_MODE_SESSION_ID = "f8feef"


def _dm_log(run_id, hypothesis_id, location, message, data=None):
    """Writes a single NDJSON debug entry for the current debug session."""
    try:
        payload = {
            "sessionId": DEBUG_MODE_SESSION_ID,
            "runId": run_id,
            "hypothesisId": hypothesis_id,
            "location": location,
            "message": message,
            "data": data,
            "timestamp": int(datetime.now().timestamp() * 1000),
        }
        with open(DEBUG_MODE_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=True) + "\n")
    except Exception:
        pass

# #endregion


def _trapz(y, x):
    """Trapezoidal integration; NumPy 2.0 renamed ``trapz`` to ``trapezoid``."""
    trap = getattr(np, "trapezoid", None) or getattr(np, "trapz")
    return trap(y, x)


def _debug_log(run_id, hypothesis_id, location, message, data):
    try:
        payload = {
            "sessionId": DEBUG_SESSION_ID,
            "runId": run_id,
            "hypothesisId": hypothesis_id,
            "location": location,
            "message": message,
            "data": data,
            "timestamp": int(datetime.now().timestamp() * 1000),
        }
        with open(DEBUG_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=True) + "\n")
    except Exception:
        pass


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
    data_ready = pyqtSignal(np.ndarray, float)  # processed data, timestamp (ALL samples for logging)
    raw_data_ready_logging = pyqtSignal(np.ndarray, float)  # unfiltered data, timestamp (ALL samples for logging)
    plot_data_ready = pyqtSignal(np.ndarray, float)  # processed data, timestamp (SUBSAMPLED for plotting)
    raw_data_ready = pyqtSignal(np.ndarray, float)  # unfiltered data, timestamp (SUBSAMPLED for plotting)
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
        
        # Buffer to accumulate samples before emitting for plotting (reduces UI load)
        self.plot_buffer = []
        self.plot_buffer_size = 20  # Emit every 20 samples (~80ms at 250Hz, increased for better performance)
        self.last_plot_time = 0
        self.plot_interval = 0.1  # Emit every 100ms maximum (increased from 40ms)
        
        # Counter for logging (emit ALL samples for logging)
        self._log_sample_counter = 0
    
    def _setup_filters(self):
        """Configures digital filters for signal processing."""
        # Bandpass filter 1-40 Hz (Butterworth, order 4) for processing/ratio
        nyquist = self.sample_rate / 2
        low = 1.0 / nyquist
        high = 40.0 / nyquist
        self.b, self.a = signal.butter(4, [low, high], btype='band')
        # Dedicated plot filter 7-13 Hz (alpha-focused live visualization)
        low_plot = 7.0 / nyquist
        high_plot = 13.0 / nyquist
        self.b_plot, self.a_plot = signal.butter(4, [low_plot, high_plot], btype='band')
        
        # Notch filter 60 Hz (to eliminate electrical line noise)
        notch_freq = 60.0
        quality_factor = 30.0
        self.b_notch, self.a_notch = signal.iirnotch(notch_freq, quality_factor, self.sample_rate)
        
        # Initial filter state (one per channel)
        zi_band_single = signal.lfilter_zi(self.b, self.a)
        zi_plot_single = signal.lfilter_zi(self.b_plot, self.a_plot)
        zi_notch_single = signal.lfilter_zi(self.b_notch, self.a_notch)
        self.zi_band = np.tile(zi_band_single[:, np.newaxis], (1, self.n_channels))
        self.zi_plot = np.tile(zi_plot_single[:, np.newaxis], (1, self.n_channels))
        self.zi_notch = np.tile(zi_notch_single[:, np.newaxis], (1, self.n_channels))
    
    def connect_to_stream(self):
        """Searches for and connects to AURA LSL stream."""
        try:
            print("Searching for AURA EEG stream...")
            # 1) Fast path: exact historical name
            streams = resolve_byprop('name', 'AURA', timeout=1.5)

            # 2) Fallback: scan all streams and match tolerant patterns
            if len(streams) == 0:
                discovered = resolve_streams(wait_time=2.0)
                candidates = []
                for s in discovered:
                    try:
                        s_name = (s.name() or "").strip()
                        s_type = (s.type() or "").strip()
                        s_sid = (s.source_id() or "").strip()
                    except Exception:
                        continue

                    name_l = s_name.lower()
                    type_l = s_type.lower()
                    sid_l = s_sid.lower()
                    # Accept common variants: "AURA", "Aura EEG", type EEG, etc.
                    if (
                        "aura" in name_l
                        or "aura" in sid_l
                        or ("eeg" in type_l and ("aura" in name_l or "brain" in name_l or "openbci" in name_l))
                    ):
                        candidates.append(s)

                streams = candidates

            if len(streams) == 0:
                # Help debugging: include what LSL streams are currently visible
                discovered = resolve_streams(wait_time=0.5)
                visible = []
                for s in discovered:
                    try:
                        visible.append(f"{s.name()} [{s.type()}]")
                    except Exception:
                        pass
                msg = "AURA stream not found"
                if visible:
                    msg += f". Visible streams: {', '.join(visible[:6])}"
                self.connection_status.emit(False, msg)
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
        bandpower = _trapz(psd[idx_band], freqs[idx_band])
        
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
                    if not hasattr(self, "_rx_counter"):
                        self._rx_counter = 0
                    self._rx_counter += 1
                    if self._rx_counter <= 30 or self._rx_counter % 200 == 0:
                        # #region agent log
                        _debug_log(
                            "pre-fix",
                            "H4",
                            "core/signal_worker.py:run",
                            "lsl_sample_received",
                            {
                                "rx_counter": self._rx_counter,
                                "sample_len": len(sample),
                                "timestamp": float(timestamp),
                            },
                        )
                        # #endregion
                    # Convert to numeric numpy array (robust against mixed LSL payload types)
                    sample_array = np.asarray(sample[:self.n_channels], dtype=float)
                    
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
                    filtered_notch = np.zeros(self.n_channels, dtype=float)
                    for i in range(self.n_channels):
                        y_notch, self.zi_notch[:, i] = signal.lfilter(
                            self.b_notch, self.a_notch, [sample_array[i]],
                            zi=self.zi_notch[:, i]
                        )
                        filtered_notch[i] = float(y_notch[0])
                    
                    # Bandpass filter
                    filtered_sample = np.zeros(self.n_channels, dtype=float)
                    for i in range(self.n_channels):
                        y_band, self.zi_band[:, i] = signal.lfilter(
                            self.b, self.a, [filtered_notch[i]],
                            zi=self.zi_band[:, i]
                        )
                        filtered_sample[i] = float(y_band[0])
                    
                    # Add 1-40 Hz processed signal to buffer (used by ratio/logging)
                    self.ring_buffer.append(filtered_sample, timestamp)
                    
                    # ============================================================
                    # LOGGING: Emit ALL samples for data logging
                    # ============================================================
                    # PyQt QueuedConnection requires a running Qt event loop in the receiver thread.
                    # Electron's eeg_bridge has no event loop, so optional direct callbacks are used there.
                    cb_data = getattr(self, "_electron_bridge_data", None)
                    if cb_data:
                        cb_data(filtered_sample, timestamp)
                    else:
                        self.data_ready.emit(filtered_sample, timestamp)
                    self.raw_data_ready_logging.emit(sample_array, timestamp)  # Raw data for logging (ALL samples)
                    if self._rx_counter <= 30 or self._rx_counter % 200 == 0:
                        # #region agent log
                        _debug_log(
                            "pre-fix",
                            "H4",
                            "core/signal_worker.py:run",
                            "sample_emitted_to_logging_signal",
                            {
                                "rx_counter": self._rx_counter,
                                "n_channels_emitted": int(len(filtered_sample)),
                                "timestamp": float(timestamp),
                            },
                        )
                        # #endregion
                    
                    # ============================================================
                    # PLOTTING: Emit only some samples to reduce UI load
                    # ============================================================
                    current_time = time.time()
                    # Build dedicated plotted signal in 7-13 Hz band.
                    filtered_plot = np.zeros(self.n_channels, dtype=float)
                    for i in range(self.n_channels):
                        y_plot, self.zi_plot[:, i] = signal.lfilter(
                            self.b_plot, self.a_plot, [filtered_notch[i]],
                            zi=self.zi_plot[:, i]
                        )
                        filtered_plot[i] = float(y_plot[0])

                    self.plot_buffer.append((sample_array, filtered_plot, timestamp))
                    
                    # Emit in batches or when maximum interval passes
                    if (len(self.plot_buffer) >= self.plot_buffer_size or 
                        (current_time - self.last_plot_time) >= self.plot_interval):
                        if self.plot_buffer:
                            # Emit the last sample from buffer for plotting
                            last_raw, last_filtered, last_ts = self.plot_buffer[-1]
                            cb_plot = getattr(self, "_electron_bridge_plot", None)
                            if cb_plot:
                                cb_plot(last_raw, last_filtered, last_ts)
                            else:
                                self.raw_data_ready.emit(last_raw, last_ts)
                                self.plot_data_ready.emit(last_filtered, last_ts)
                            self.plot_buffer.clear()
                            self.last_plot_time = current_time
                
            except Exception as e:
                # #region debug_acquisition_exception
                if not hasattr(self, "_dm_acq_err_count"):
                    self._dm_acq_err_count = 0
                if self._dm_acq_err_count < 5:
                    _dm_log(
                        "pre_ui_mac",
                        "H_ACQUISITION_EXCEPT",
                        "core/signal_worker.py:run",
                        "acquisition_exception",
                        {"error_type": type(e).__name__, "error": str(e)},
                    )
                    self._dm_acq_err_count += 1
                # #endregion
                print(f"Acquisition error: {str(e)}")
                time.sleep(0.01)  # Small pause to avoid infinite loops
    
    def stop(self):
        """Stops data acquisition."""
        self.running = False
        print("Stopping data acquisition...")

