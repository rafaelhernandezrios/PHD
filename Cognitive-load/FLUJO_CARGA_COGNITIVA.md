# Processing Flow: EEG Acquisition to Cognitive Load Calculation

## General Description

This document describes the complete flow of EEG signal processing from AURA device acquisition to real-time cognitive load index calculation.

---

## 1. Data Acquisition (LSL Stream)

### 1.1 Connection to AURA Device

**File:** `signal_worker.py` - Method `connect_to_stream()`

- **Protocol:** Lab Streaming Layer (LSL)
- **Stream search:** `resolve_byprop('name', 'AURA', timeout=1.0)`
- **Device configuration:**
  - **Stream name:** "AURA"
  - **Channels:** 8 EEG channels
  - **Sampling rate:** 250 Hz (250 samples per second)
  - **Data format:** Values in nanovolts (nV)

### 1.2 Continuous Acquisition

**File:** `signal_worker.py` - Method `run()`

The main loop acquires data continuously:

```python
while self.running:
    sample, timestamp = self.inlet.pull_sample(timeout=0.1)
```

**Characteristics:**
- **Frequency:** 250 samples/second (4 ms between samples)
- **Sample format:** List with 8 values (one per channel)
- **Typical values:** Range from -70,000 to -156,000 nV (nanovolts)
- **Timestamp:** Absolute time from LSL system

---

## 2. Signal Preprocessing

### 2.1 Data Conversion

**File:** `signal_worker.py` - Line 229

```python
sample_array = np.array(sample[:self.n_channels])
```

- Converts Python list to NumPy array
- Extracts first 8 values (one per channel)
- Resulting shape: `(8,)` - 1D array with 8 elements

### 2.2 Digital Filtering

**File:** `signal_worker.py` - Lines 244-259

Two filters are applied in cascade to each sample:

#### A. Notch Filter (60 Hz)
- **Purpose:** Eliminate electrical line noise (50/60 Hz)
- **Type:** IIR Notch filter
- **Central frequency:** 60 Hz
- **Quality factor (Q):** 30.0
- **Implementation:** `signal.iirnotch(60.0, 30.0, 250.0)`

#### B. Bandpass Filter (1-40 Hz)
- **Purpose:** Eliminate frequency components outside the relevant EEG range
- **Type:** Butterworth order 4
- **Passband:** 1-40 Hz
- **Implementation:** `signal.butter(4, [low, high], btype='band')`
  - `low = 1.0 / nyquist` (normalized)
  - `high = 40.0 / nyquist` (normalized)
  - `nyquist = sample_rate / 2 = 125 Hz`

**Processing:**
- Filters are applied **sample by sample** (real-time filtering)
- Internal state (`zi_band`, `zi_notch`) is maintained for each channel
- This allows causal filtering without needing a previous buffer

**Result:** Filtered signal ready for spectral analysis

---

## 3. Circular Buffer Storage

### 3.1 RingBuffer

**File:** `signal_worker.py` - Class `RingBuffer`

**Characteristics:**
- **Type:** Circular buffer (FIFO)
- **Size:** `buffer_samples * 2 = 500 * 2 = 1000 samples`
- **Duration:** ~4 seconds of data (1000 samples / 250 Hz)
- **Structure:** NumPy array of shape `(1000, 8)`

**Operations:**
- `append(data, timestamp)`: Adds new sample
- `get_window(window_samples)`: Gets moving window of 2 seconds (500 samples)

### 3.2 Filtered Data Storage

**File:** `signal_worker.py` - Line 262

```python
self.ring_buffer.append(filtered_sample, timestamp)
```

- Stores **filtered** data (not raw)
- Maintains last 4 seconds of data
- Allows spectral analysis calculation in moving windows

---

## 4. Performance Optimization

### 4.1 Emission Buffer

**File:** `signal_worker.py` - Lines 264-277

To avoid saturating Qt's event queue:

- **Accumulation:** Up to 20 samples accumulated before emitting
- **Maximum interval:** Emits every 100ms maximum (even with fewer samples)
- **Effective frequency:** ~10 Hz instead of 250 Hz

**Result:** Reduces load on graphical interface without losing significant information

### 4.2 Emitted Signals

**File:** `signal_worker.py` - Lines 274-275

```python
self.raw_data_ready.emit(last_raw, last_ts)      # Unfiltered data
self.data_ready.emit(last_filtered, last_ts)    # Filtered data
```

- `raw_data_ready`: For raw signal visualization
- `data_ready`: For logging and analysis

---

## 5. Raw Signal Visualization

### 5.1 Reception in UI

**File:** `ui_main.py` - Method `update_raw_plot()`

**Process:**
1. Receives raw (unfiltered) data via PyQt signal
2. Applies subsampling: updates every 5 received samples
3. Converts nanovolts to microvolts: `values_microvolts = raw_values / 1000.0`

### 5.2 Scaling and Offset

**File:** `ui_main.py` - Lines 437-450

**Unit conversion:**
- **Input:** Nanovolts (nV) - typical range: -70,000 to -156,000 nV
- **Output:** Microvolts (μV) - typical range: -70 to -156 μV
- **Formula:** `μV = nV / 1000`

**Channel separation:**
- **Offset per channel:** 200 μV between each channel
- **Channel 0:** Offset = 0 μV
- **Channel 1:** Offset = 200 μV
- **Channel 2:** Offset = 400 μV
- ...
- **Channel 7:** Offset = 1400 μV

**Visualization:**
- Each channel is plotted with a different color
- Channels are separated vertically to avoid overlap
- Y-axis range of plot: -300 to 1500 μV

---

## 6. Spectral Bandpower Calculation

### 6.1 Temporal Window Extraction

**File:** `signal_worker.py` - Method `get_cognitive_load_ratio()`

```python
window_data = self.ring_buffer.get_window(self.buffer_samples)
```

- **Window size:** 500 samples = 2 seconds (at 250 Hz)
- **Data:** Filtered signal (1-40 Hz, without 60 Hz)
- **Shape:** `(500, 8)` - 500 samples × 8 channels

### 6.2 Channel Selection

**File:** `signal_worker.py` - Lines 196-198

```python
fz_signal = window_data[:, self.fz_channel]  # Channel 0 (Fz - frontal)
pz_signal = window_data[:, self.pz_channel]  # Channel 4 (Pz - parietal)
```

**Channel mapping:**
- **Channel 0 (Fz):** Frontal electrode - used for Theta band
- **Channel 4 (Pz):** Parietal electrode - used for Alpha band

### 6.3 Welch's Method for Spectral Estimation

**File:** `signal_worker.py` - Method `calculate_bandpower()`

**Parameters:**
- **Method:** Welch's periodogram
- **Window:** `nperseg = min(len(signal_data), sample_rate) = 250 samples`
- **Overlap:** `noverlap = nperseg // 2 = 125 samples`
- **Frequency resolution:** ~1 Hz

**Process:**
1. Divides signal into overlapping segments
2. Calculates FFT of each segment
3. Averages periodograms
4. Obtains Power Spectral Density (PSD) in μV²/Hz

**Result:** `freqs, psd` - Frequencies and power spectral density

### 6.4 Band Power Calculation

**File:** `signal_worker.py` - Lines 177-180

**Frequency bands:**

#### Theta Band (4-7 Hz)
- **Channel:** Fz (frontal)
- **Range:** 4.0 to 7.0 Hz
- **Calculation:**
  ```python
  idx_band = np.logical_and(freqs >= 4.0, freqs <= 7.0)
  theta_power = np.trapz(psd[idx_band], freqs[idx_band])
  ```
- **Units:** μV² (integral of PSD over the band)

#### Alpha Band (8-12 Hz)
- **Channel:** Pz (parietal)
- **Range:** 8.0 to 12.0 Hz
- **Calculation:**
  ```python
  idx_band = np.logical_and(freqs >= 8.0, freqs <= 12.0)
  alpha_power = np.trapz(psd[idx_band], freqs[idx_band])
  ```
- **Units:** μV² (integral of PSD over the band)

**Integration method:** Trapezoidal rule (`np.trapz`) to calculate the area under the PSD curve in each band.

---

## 7. Cognitive Load Index Calculation

### 7.1 Ratio Formula

**File:** `signal_worker.py` - Lines 207-209

```python
ratio = theta_power / alpha_power
```

**Mathematical formula:**

\[
\text{Cognitive Load Ratio} = \frac{\text{Theta Power}_{Fz}}{\text{Alpha Power}_{Pz}}
\]

### 7.2 Interpretation

**Ratio values:**
- **Ratio > 1:** Higher relative Theta power → **Higher cognitive load**
- **Ratio < 1:** Higher relative Alpha power → **Lower cognitive load**
- **Ratio ≈ 1:** Moderate cognitive load

**Neurophysiological justification:**
- **Theta (4-7 Hz) in Fz:** Associated with mental effort, sustained attention, working memory
- **Alpha (8-12 Hz) in Pz:** Associated with relaxation, passive processing, rest state
- **Theta/Alpha Ratio:** Robust indicator of cognitive load in attention tasks

### 7.3 Validation

**File:** `signal_worker.py` - Line 207

```python
if alpha_power > 0:
    ratio = theta_power / alpha_power
    return ratio, theta_power, alpha_power
```

- Verifies that `alpha_power > 0` to avoid division by zero
- Returns `None` if there's not enough data or alpha_power is zero

---

## 8. Real-Time Update

### 8.1 Calculation Timer

**File:** `main.py` - Lines 68-70

```python
self.ratio_timer = QTimer()
self.ratio_timer.timeout.connect(self.calculate_and_update_ratio)
self.ratio_timer.start(1000)  # Every 1 second
```

**Update frequency:**
- **Ratio calculation:** Every 1 second
- **Analysis window:** 2 seconds (500 samples)
- **Overlap:** 50% between consecutive windows

### 8.2 Update Flow

**File:** `main.py` - Method `calculate_and_update_ratio()`

1. **Timer triggers** every 1 second
2. **Calls** `signal_worker.get_cognitive_load_ratio()`
3. **Gets** 2-second window from circular buffer
4. **Calculates** Theta and Alpha bandpower
5. **Calculates** ratio = Theta / Alpha
6. **Emits** result to UI for visualization

### 8.3 Ratio Visualization

**File:** `ui_main.py` - Method `update_ratio_plot()`

- **Plot:** Temporal line of ratio
- **Update:** Every time a new ratio is calculated
- **Buffer:** Last 300 points (~5 minutes at 1 Hz)
- **X-axis:** Relative time (seconds from present)
- **Y-axis:** Ratio value (dimensionless)

---

## 9. Complete Flow Summary

```
┌─────────────────────────────────────────────────────────────┐
│ 1. LSL ACQUISITION                                           │
│    AURA → LSL Stream → pull_sample()                         │
│    Frequency: 250 Hz                                          │
│    Format: 8 channels in nanovolts                           │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. PREPROCESSING                                             │
│    • Conversion to NumPy array                                │
│    • Notch filter 60 Hz                                       │
│    • Bandpass filter 1-40 Hz                                │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. STORAGE                                                   │
│    • Circular RingBuffer (1000 samples = 4 sec)            │
│    • Maintains filter state                                  │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ├─────────────────┐
                       ▼                 ▼
        ┌──────────────────────┐  ┌──────────────────────┐
        │ 4. VISUALIZATION     │  │ 5. ANALYSIS          │
        │    Raw signals       │  │    Spectral          │
        │    (8 channels)      │  │    • Welch PSD       │
        └──────────────────────┘  │    • Bandpower        │
                                  └──────────┬───────────┘
                                             │
                                             ▼
                                  ┌──────────────────────┐
                                  │ 6. RATIO CALCULATION │
                                  │    Theta_Fz /        │
                                  │    Alpha_Pz           │
                                  └──────────┬───────────┘
                                             │
                                             ▼
                                  ┌──────────────────────┐
                                  │ 7. VISUALIZATION     │
                                  │    Temporal ratio     │
                                  │    (update            │
                                  │     every 1 second)   │
                                  └──────────────────────┘
```

---

## 10. Key Technical Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| **Sampling rate** | 250 Hz | Samples per second from device |
| **Number of channels** | 8 | Simultaneous EEG channels |
| **Notch Filter** | 60 Hz, Q=30 | Electrical line noise elimination |
| **Bandpass Filter** | 1-40 Hz, order 4 | EEG frequency range |
| **Analysis window** | 2 seconds (500 samples) | For bandpower calculation |
| **Theta Band** | 4-7 Hz | Channel Fz (frontal) |
| **Alpha Band** | 8-12 Hz | Channel Pz (parietal) |
| **Calculation frequency** | 1 Hz | Ratio updates every second |
| **Spectral method** | Welch's periodogram | PSD estimation |
| **Welch window** | 250 samples (1 second) | Segment size |
| **Welch overlap** | 50% (125 samples) | Between segments |

---

## 11. Performance Considerations

### 11.1 Implemented Optimizations

1. **Emission buffer:** Reduces signals from 250 Hz to ~10 Hz
2. **UI subsampling:** Updates plots every 5 samples
3. **Logging subsampling:** Saves every 5 samples (50 Hz)
4. **Adjusted timers:** Plots at ~5 FPS, ratio at 1 Hz

### 11.2 Memory Usage

- **RingBuffer:** ~32 KB (1000 samples × 8 channels × 4 bytes)
- **Plot buffers:** ~40 KB per plot
- **Logging:** Depends on experiment duration (subsampled)

---

## 12. Technical References

- **LSL (Lab Streaming Layer):** Biomedical data streaming protocol
- **Welch's Method:** Welch, P. D. (1967). "The use of fast Fourier transform for the estimation of power spectra"
- **Digital Filters:** Oppenheim & Schafer, "Discrete-Time Signal Processing"
- **EEG Bands:** Klimesch, W. (1999). "EEG alpha and theta oscillations reflect cognitive and memory performance"

---

**Last update:** December 2025 
**System version:** 1.0
