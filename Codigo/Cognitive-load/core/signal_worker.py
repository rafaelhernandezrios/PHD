"""
signal_worker.py
QThread class for EEG data acquisition via LSL and real-time signal processing.
Handles filtering, circular buffer and bandpower calculation.

2026-08 revision (ERP-ready acquisition):
  * chunked acquisition (``pull_chunk``) + vectorised filtering, so the loop
    keeps up with 250 Hz x 8 ch while writing raw AND filtered to disk;
  * inlet built with LSL post-processing (clocksync/dejitter/monotonize) so
    timestamps are monotonic and epochable;
  * filter states primed from the first sample, so the start-up transient is
    no longer written into the recording;
  * raw samples reach the Electron bridge (previously emitted into a Qt signal
    that nothing was connected to);
  * live acquisition-health metrics: effective sampling rate, dropped-sample
    estimate, and flat/rail channel detection on the RAW signal.
"""

import numpy as np
from scipy import signal
from PyQt5.QtCore import QThread, pyqtSignal
from pylsl import StreamInlet, resolve_byprop, resolve_streams, local_clock
from collections import deque
import time

# LSL post-processing flags. Guarded because very old pylsl builds lack them.
try:
    from pylsl import proc_clocksync, proc_dejitter, proc_monotonize
    LSL_PROC_FLAGS = proc_clocksync | proc_dejitter | proc_monotonize
except ImportError:  # pragma: no cover - depends on pylsl build
    LSL_PROC_FLAGS = 0


# Raw-signal health thresholds, expressed in AURA ADC counts.
# A disconnected electrode rails at ~-375000 with a standard deviation of
# exactly 0; after the band-pass it looks like the cleanest channel on screen,
# which is precisely why it has to be checked on the raw signal.
RAIL_ABS_THRESHOLD = 3.0e5
FLAT_STD_THRESHOLD = 1.0


def _trapz(y, x):
    """Trapezoidal integration; NumPy 2.0 renamed ``trapz`` to ``trapezoid``."""
    trap = getattr(np, "trapezoid", None) or getattr(np, "trapz")
    return trap(y, x)


def make_inlet(stream_info, max_buflen=360):
    """
    Builds a StreamInlet with LSL post-processing enabled.

    ``proc_clocksync``  - corrects for the offset between sender and receiver clocks
    ``proc_dejitter``   - fits a linear model to smooth out timestamp jitter
    ``proc_monotonize`` - guarantees timestamps never go backwards

    Without these flags, recordings from this rig showed 5-11% of consecutive
    timestamp deltas being zero or negative, which makes stimulus-locked
    epoching impossible.
    """
    return StreamInlet(
        stream_info,
        max_buflen=max_buflen,
        processing_flags=LSL_PROC_FLAGS,
    )


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

    def append_chunk(self, data, timestamps=None):
        """
        Adds a block of samples at once.

        Args:
            data: Array of shape (n_samples, n_channels)
            timestamps: Optional sequence of timestamps, one per sample
        """
        n = len(data)
        if n == 0:
            return
        if n >= self.maxlen:
            # Chunk larger than the buffer: keep only the newest tail.
            data = data[-self.maxlen:]
            if timestamps is not None:
                timestamps = timestamps[-self.maxlen:]
            n = self.maxlen

        end = self.write_idx + n
        if end <= self.maxlen:
            self.buffer[self.write_idx:end] = data
        else:
            split = self.maxlen - self.write_idx
            self.buffer[self.write_idx:] = data[:split]
            self.buffer[:end - self.maxlen] = data[split:]

        if timestamps is not None:
            self.timestamps.extend(timestamps)

        if end >= self.maxlen:
            self.is_full = True
        self.write_idx = end % self.maxlen

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

    def __init__(self, sample_rate=250, n_channels=8, buffer_duration=2.0,
                 line_freq=60.0):
        """
        Args:
            sample_rate: Sampling rate in Hz (250 Hz for AURA)
            n_channels: Number of EEG channels (8 for AURA)
            buffer_duration: Buffer duration in seconds (2.0 s for FFT window)
            line_freq: Mains frequency for the notch filter (60 Hz in Mexico,
                50 Hz in most of Europe/Asia)
        """
        super().__init__()
        self.sample_rate = sample_rate
        self.n_channels = n_channels
        self.line_freq = line_freq
        self.buffer_samples = int(buffer_duration * sample_rate)

        # Circular buffers: filtered (used for the cognitive-load ratio) and
        # raw (used for electrode-health checks, which must NOT see the filter).
        self.ring_buffer = RingBuffer(maxlen=self.buffer_samples * 2, n_channels=n_channels)
        self.raw_ring_buffer = RingBuffer(maxlen=self.buffer_samples * 2, n_channels=n_channels)

        # Filters
        self._setup_filters()

        # Thread control
        self.running = False
        self.inlet = None

        # Channel indices for analysis (Fz = channel 3, Pz = channel 6)
        # Channel mapping: 0=Fp1, 1=Fp2, 2=F3, 3=Fz, 4=F4, 5=P3, 6=Pz, 7=P4
        # NOTE: verify this against the AURA montage before every recording.
        self.fz_channel = 3
        self.pz_channel = 6

        # Plot throttling (the UI only needs ~10 frames per second)
        self.last_plot_time = 0.0
        self.plot_interval = 0.1

        # Acquisition health bookkeeping
        self._rx_total = 0
        self._rate_window = deque()      # (wall_clock, cumulative_samples)
        self._rate_window_seconds = 5.0
        self._first_lsl_ts = None
        self._last_lsl_ts = None
        self._empty_pulls = 0
        self._stream_lost = False
        self._filters_primed = False

        # Optional direct callbacks used by the Electron bridge, which runs
        # without a Qt event loop and therefore cannot receive queued signals.
        self._electron_bridge_chunk = None   # (raw_block, filt_block, ts_block)
        self._electron_bridge_plot = None    # (raw_sample, plot_sample, ts)
        self._electron_bridge_status = None  # (dict) health / stream-loss events

    # ------------------------------------------------------------------
    # Filter setup
    # ------------------------------------------------------------------
    def _setup_filters(self):
        """Configures digital filters for signal processing."""
        nyquist = self.sample_rate / 2

        # Band-pass 1-40 Hz (Butterworth, order 4) for processing/ratio
        self.b, self.a = signal.butter(4, [1.0 / nyquist, 40.0 / nyquist], btype='band')

        # Dedicated plot filter. Deliberately WIDE (0.5-45 Hz): a narrow alpha
        # filter hides drift, pops, EMG and dead electrodes, which is exactly
        # what the operator needs to see while placing the cap.
        self.b_plot, self.a_plot = signal.butter(4, [0.5 / nyquist, 45.0 / nyquist], btype='band')

        # Notch at the local mains frequency
        self.b_notch, self.a_notch = signal.iirnotch(self.line_freq, 30.0, self.sample_rate)

        # Per-channel filter state; primed on the first chunk from the actual
        # first sample (see _prime_filters). Priming with the unscaled output
        # of lfilter_zi assumes an initial input of 1.0, but AURA delivers raw
        # ADC counts of order 1e5, which produced a multi-second transient that
        # was written straight into the CSV.
        self.zi_band = np.zeros((max(len(self.a), len(self.b)) - 1, self.n_channels))
        self.zi_plot = np.zeros((max(len(self.a_plot), len(self.b_plot)) - 1, self.n_channels))
        self.zi_notch = np.zeros((max(len(self.a_notch), len(self.b_notch)) - 1, self.n_channels))
        self._filters_primed = False

    def _prime_filters(self, first_sample):
        """Sets initial filter states from the first observed sample."""
        zi_notch = signal.lfilter_zi(self.b_notch, self.a_notch)
        self.zi_notch = zi_notch[:, None] * first_sample[None, :]

        # The notch has unity DC gain, so the band-pass sees roughly the same
        # starting value.
        zi_band = signal.lfilter_zi(self.b, self.a)
        self.zi_band = zi_band[:, None] * first_sample[None, :]

        zi_plot = signal.lfilter_zi(self.b_plot, self.a_plot)
        self.zi_plot = zi_plot[:, None] * first_sample[None, :]

        self._filters_primed = True

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------
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

            self.inlet = make_inlet(streams[0])
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

    # ------------------------------------------------------------------
    # Spectral analysis
    # ------------------------------------------------------------------
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
        freqs, psd = signal.welch(signal_data, sample_rate, nperseg=nperseg, noverlap=nperseg // 2)

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

    # ------------------------------------------------------------------
    # Acquisition health
    # ------------------------------------------------------------------
    def get_acquisition_health(self):
        """
        Live acquisition quality report.

        Returns a dict with the effective sampling rate over the last few
        seconds, an estimate of how many samples were lost, and a per-channel
        verdict computed on the RAW signal (flat / rail / ok).
        """
        # Effective rate over the trailing window, measured on LSL timestamps
        # rather than arrival times. Delivery is bursty (the amplifier sends
        # chunks, the OS schedules us when it feels like it), so arrival times
        # produce wild readings; LSL timestamps track real acquisition time.
        # Sample loss still shows up correctly: fewer samples across the same
        # timestamp span means a lower rate.
        effective_fs = 0.0
        if len(self._rate_window) >= 2:
            t0, n0 = self._rate_window[0]
            t1, n1 = self._rate_window[-1]
            if t1 > t0:
                effective_fs = (n1 - n0) / (t1 - t0)

        # Expected vs received over the whole session, using LSL timestamps
        expected = 0
        if self._first_lsl_ts is not None and self._last_lsl_ts is not None:
            span = self._last_lsl_ts - self._first_lsl_ts
            if span > 0:
                expected = int(span * self.sample_rate)
        dropped = max(0, expected - self._rx_total) if expected else 0
        drop_pct = (100.0 * dropped / expected) if expected else 0.0

        channels = []
        window = self.raw_ring_buffer.get_window(min(self.buffer_samples, self.sample_rate))
        if len(window) >= 20:
            for i in range(self.n_channels):
                col = window[:, i]
                std = float(np.std(col))
                mean = float(np.mean(col))
                absmax = float(np.max(np.abs(col)))
                if absmax > RAIL_ABS_THRESHOLD:
                    status = "rail"
                elif std < FLAT_STD_THRESHOLD:
                    status = "flat"
                else:
                    status = "ok"
                channels.append({"index": i, "std": std, "mean": mean, "status": status})

        return {
            "effective_fs": round(effective_fs, 2),
            "nominal_fs": float(self.sample_rate),
            "samples_total": int(self._rx_total),
            "dropped_estimate": int(dropped),
            "dropped_pct": round(drop_pct, 2),
            "stream_lost": bool(self._stream_lost),
            "warming_up": bool(self.is_warming_up()),
            "channels": channels,
        }

    def _note_received(self, n_samples, timestamps):
        """Updates health counters after a chunk arrives."""
        self._rx_total += n_samples

        if timestamps:
            if self._first_lsl_ts is None:
                self._first_lsl_ts = timestamps[0]
            self._last_lsl_ts = timestamps[-1]
            now_ts = timestamps[-1]
            self._rate_window.append((now_ts, self._rx_total))
            while self._rate_window and (now_ts - self._rate_window[0][0]) > self._rate_window_seconds:
                self._rate_window.popleft()

    def is_warming_up(self):
        """True until enough data has accumulated for the health check to mean anything."""
        if self._first_lsl_ts is None or self._last_lsl_ts is None:
            return True
        return (self._last_lsl_ts - self._first_lsl_ts) < 2.0

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def run(self):
        """Main thread loop. Acquires and processes data continuously."""
        if not self.inlet:
            if not self.connect_to_stream():
                return

        self.running = True
        self._stream_lost = False
        self._empty_pulls = 0

        # Drop whatever accumulated in the inlet between resolving the stream
        # and starting the loop. Without this, the first pull returns a burst
        # of stale samples that skews the rate estimate and the filter priming.
        try:
            self.inlet.flush()
        except Exception:
            pass

        print("Starting data acquisition...")

        # Pull at ~20 Hz: large enough to amortise Python overhead, small
        # enough that the live plot and the phase labels stay responsive.
        max_chunk = max(1, int(self.sample_rate * 0.2))

        while self.running:
            try:
                samples, timestamps = self.inlet.pull_chunk(
                    timeout=0.2, max_samples=max_chunk
                )

                if not samples:
                    # An empty pull is normal once in a while; a long run of
                    # them means the amplifier or the LSL outlet went away.
                    self._empty_pulls += 1
                    if self._empty_pulls >= 10 and not self._stream_lost:
                        self._stream_lost = True
                        msg = "EEG stream stopped delivering samples."
                        self.connection_status.emit(False, msg)
                        cb_status = self._electron_bridge_status
                        if cb_status:
                            cb_status({"stream_lost": True, "message": msg})
                    continue

                if self._stream_lost:
                    self._stream_lost = False
                    cb_status = self._electron_bridge_status
                    if cb_status:
                        cb_status({"stream_lost": False, "message": "Stream recovered."})
                self._empty_pulls = 0

                # (n_samples, n_channels), numeric and trimmed to the montage
                raw_block = np.asarray(samples, dtype=float)
                if raw_block.ndim == 1:
                    raw_block = raw_block.reshape(1, -1)
                raw_block = raw_block[:, :self.n_channels]
                if raw_block.shape[1] < self.n_channels:
                    pad = self.n_channels - raw_block.shape[1]
                    raw_block = np.hstack([raw_block, np.full((len(raw_block), pad), np.nan)])

                if not self._filters_primed:
                    self._prime_filters(raw_block[0])

                # Vectorised filtering: one lfilter call per stage for the whole
                # block and all channels, instead of one call per sample per
                # channel per stage (which was 6000 calls/s at 250 Hz).
                notch_block, self.zi_notch = signal.lfilter(
                    self.b_notch, self.a_notch, raw_block, axis=0, zi=self.zi_notch
                )
                filt_block, self.zi_band = signal.lfilter(
                    self.b, self.a, notch_block, axis=0, zi=self.zi_band
                )
                plot_block, self.zi_plot = signal.lfilter(
                    self.b_plot, self.a_plot, notch_block, axis=0, zi=self.zi_plot
                )

                self.ring_buffer.append_chunk(filt_block, timestamps)
                self.raw_ring_buffer.append_chunk(raw_block, timestamps)
                self._note_received(len(raw_block), timestamps)

                # ---------- LOGGING: every sample, raw AND filtered ----------
                cb_chunk = self._electron_bridge_chunk
                if cb_chunk:
                    cb_chunk(raw_block, filt_block, timestamps)
                else:
                    # Legacy PyQt path (apps/main.py, apps/main_baseline.py):
                    # these consumers expect one signal emission per sample.
                    for i in range(len(raw_block)):
                        ts = float(timestamps[i])
                        self.data_ready.emit(filt_block[i], ts)
                        self.raw_data_ready_logging.emit(raw_block[i], ts)

                # ---------- PLOTTING: throttled to ~10 Hz ----------
                current_time = time.time()
                if (current_time - self.last_plot_time) >= self.plot_interval:
                    last_raw = raw_block[-1]
                    last_plot = plot_block[-1]
                    last_ts = float(timestamps[-1])
                    cb_plot = self._electron_bridge_plot
                    if cb_plot:
                        cb_plot(last_raw, last_plot, last_ts)
                    else:
                        self.raw_data_ready.emit(last_raw, last_ts)
                        self.plot_data_ready.emit(last_plot, last_ts)
                    self.last_plot_time = current_time

            except Exception as e:
                print(f"Acquisition error: {str(e)}")
                cb_status = self._electron_bridge_status
                if cb_status:
                    cb_status({"error": f"{type(e).__name__}: {e}"})
                time.sleep(0.01)  # Small pause to avoid infinite loops

    def stop(self):
        """Stops data acquisition."""
        self.running = False
        print("Stopping data acquisition...")

    @staticmethod
    def lsl_clock():
        """Current LSL clock value, for aligning stimulus markers to the EEG."""
        return float(local_clock())
