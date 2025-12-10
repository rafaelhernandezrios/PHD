# Flujo de Procesamiento: Adquisición EEG hasta Cálculo de Carga Cognitiva

## Descripción General

Este documento describe el flujo completo de procesamiento de señales EEG desde la adquisición del dispositivo AURA hasta el cálculo del índice de carga cognitiva en tiempo real.

---

## 1. Adquisición de Datos (LSL Stream)

### 1.1 Conexión al Dispositivo AURA

**Archivo:** `signal_worker.py` - Método `connect_to_stream()`

- **Protocolo:** Lab Streaming Layer (LSL)
- **Búsqueda del stream:** `resolve_byprop('name', 'AURA', timeout=1.0)`
- **Configuración del dispositivo:**
  - **Nombre del stream:** "AURA"
  - **Canales:** 8 canales EEG
  - **Tasa de muestreo:** 250 Hz (250 muestras por segundo)
  - **Formato de datos:** Valores en nanovolts (nV)

### 1.2 Adquisición Continua

**Archivo:** `signal_worker.py` - Método `run()`

El loop principal adquiere datos continuamente:

```python
while self.running:
    sample, timestamp = self.inlet.pull_sample(timeout=0.1)
```

**Características:**
- **Frecuencia:** 250 muestras/segundo (4 ms entre muestras)
- **Formato de sample:** Lista con 8 valores (uno por canal)
- **Valores típicos:** Rango de -70,000 a -156,000 nV (nanovolts)
- **Timestamp:** Tiempo absoluto del sistema LSL

---

## 2. Preprocesamiento de Señal

### 2.1 Conversión de Datos

**Archivo:** `signal_worker.py` - Línea 229

```python
sample_array = np.array(sample[:self.n_channels])
```

- Convierte la lista de Python a array NumPy
- Extrae los primeros 8 valores (uno por canal)
- Shape resultante: `(8,)` - array 1D con 8 elementos

### 2.2 Filtrado Digital

**Archivo:** `signal_worker.py` - Líneas 244-259

Se aplican dos filtros en cascada a cada muestra:

#### A. Filtro Notch (60 Hz)
- **Propósito:** Eliminar ruido de línea eléctrica (50/60 Hz)
- **Tipo:** IIR Notch filter
- **Frecuencia central:** 60 Hz
- **Factor de calidad (Q):** 30.0
- **Implementación:** `signal.iirnotch(60.0, 30.0, 250.0)`

#### B. Filtro Pasabanda (1-40 Hz)
- **Propósito:** Eliminar componentes de frecuencia fuera del rango EEG relevante
- **Tipo:** Butterworth de orden 4
- **Banda de paso:** 1-40 Hz
- **Implementación:** `signal.butter(4, [low, high], btype='band')`
  - `low = 1.0 / nyquist` (normalizado)
  - `high = 40.0 / nyquist` (normalizado)
  - `nyquist = sample_rate / 2 = 125 Hz`

**Procesamiento:**
- Los filtros se aplican **muestra por muestra** (filtrado en tiempo real)
- Se mantiene el estado interno (`zi_band`, `zi_notch`) para cada canal
- Esto permite filtrado causal sin necesidad de buffer previo

**Resultado:** Señal filtrada lista para análisis espectral

---

## 3. Almacenamiento en Buffer Circular

### 3.1 RingBuffer

**Archivo:** `signal_worker.py` - Clase `RingBuffer`

**Características:**
- **Tipo:** Buffer circular (FIFO)
- **Tamaño:** `buffer_samples * 2 = 500 * 2 = 1000 muestras`
- **Duración:** ~4 segundos de datos (1000 muestras / 250 Hz)
- **Estructura:** Array NumPy de shape `(1000, 8)`

**Operaciones:**
- `append(data, timestamp)`: Añade nueva muestra
- `get_window(window_samples)`: Obtiene ventana móvil de 2 segundos (500 muestras)

### 3.2 Almacenamiento de Datos Filtrados

**Archivo:** `signal_worker.py` - Línea 262

```python
self.ring_buffer.append(filtered_sample, timestamp)
```

- Almacena los datos **filtrados** (no los raw)
- Mantiene los últimos 4 segundos de datos
- Permite cálculo de análisis espectral en ventanas móviles

---

## 4. Optimización de Rendimiento

### 4.1 Buffer de Emisión

**Archivo:** `signal_worker.py` - Líneas 264-277

Para evitar saturar la cola de eventos de Qt:

- **Acumulación:** Se acumulan hasta 10 muestras antes de emitir
- **Intervalo máximo:** Emite cada 40ms máximo (incluso con menos muestras)
- **Frecuencia efectiva:** ~25 Hz en lugar de 250 Hz

**Resultado:** Reduce la carga en la interfaz gráfica sin perder información significativa

### 4.2 Señales Emitidas

**Archivo:** `signal_worker.py` - Líneas 274-275

```python
self.raw_data_ready.emit(last_raw, last_ts)      # Datos sin filtrar
self.data_ready.emit(last_filtered, last_ts)    # Datos filtrados
```

- `raw_data_ready`: Para visualización de señales raw
- `data_ready`: Para logging y análisis

---

## 5. Visualización de Señales Raw

### 5.1 Recepción en UI

**Archivo:** `ui_main.py` - Método `update_raw_plot()`

**Proceso:**
1. Recibe datos raw (sin filtrar) vía señal PyQt
2. Aplica submuestreo: actualiza cada 2 muestras recibidas
3. Convierte nanovolts a microvolts: `values_microvolts = raw_values / 1000.0`

### 5.2 Escalado y Offset

**Archivo:** `ui_main.py` - Líneas 437-450

**Conversión de unidades:**
- **Entrada:** Nanovolts (nV) - rango típico: -70,000 a -156,000 nV
- **Salida:** Microvolts (μV) - rango típico: -70 a -156 μV
- **Fórmula:** `μV = nV / 1000`

**Separación de canales:**
- **Offset por canal:** 200 μV entre cada canal
- **Canal 0:** Offset = 0 μV
- **Canal 1:** Offset = 200 μV
- **Canal 2:** Offset = 400 μV
- ...
- **Canal 7:** Offset = 1400 μV

**Visualización:**
- Cada canal se grafica con un color diferente
- Los canales están separados verticalmente para evitar solapamiento
- Rango Y del gráfico: -300 a 1500 μV

---

## 6. Cálculo de Bandpower Espectral

### 6.1 Extracción de Ventana Temporal

**Archivo:** `signal_worker.py` - Método `get_cognitive_load_ratio()`

```python
window_data = self.ring_buffer.get_window(self.buffer_samples)
```

- **Tamaño de ventana:** 500 muestras = 2 segundos (a 250 Hz)
- **Datos:** Señal filtrada (1-40 Hz, sin 60 Hz)
- **Shape:** `(500, 8)` - 500 muestras × 8 canales

### 6.2 Selección de Canales

**Archivo:** `signal_worker.py` - Líneas 196-198

```python
fz_signal = window_data[:, self.fz_channel]  # Canal 0 (Fz - frontal)
pz_signal = window_data[:, self.pz_channel]  # Canal 4 (Pz - parietal)
```

**Mapeo de canales:**
- **Canal 0 (Fz):** Electrodo frontal - usado para banda Theta
- **Canal 4 (Pz):** Electrodo parietal - usado para banda Alpha

### 6.3 Método de Welch para Estimación Espectral

**Archivo:** `signal_worker.py` - Método `calculate_bandpower()`

**Parámetros:**
- **Método:** Welch's periodogram
- **Ventana:** `nperseg = min(len(signal_data), sample_rate) = 250 muestras`
- **Solapamiento:** `noverlap = nperseg // 2 = 125 muestras`
- **Resolución frecuencial:** ~1 Hz

**Proceso:**
1. Divide la señal en segmentos solapados
2. Calcula FFT de cada segmento
3. Promedia los periodogramas
4. Obtiene Power Spectral Density (PSD) en μV²/Hz

**Resultado:** `freqs, psd` - Frecuencias y densidad espectral de potencia

### 6.4 Cálculo de Potencia en Bandas

**Archivo:** `signal_worker.py` - Líneas 177-180

**Bandas de frecuencia:**

#### Banda Theta (4-7 Hz)
- **Canal:** Fz (frontal)
- **Rango:** 4.0 a 7.0 Hz
- **Cálculo:**
  ```python
  idx_band = np.logical_and(freqs >= 4.0, freqs <= 7.0)
  theta_power = np.trapz(psd[idx_band], freqs[idx_band])
  ```
- **Unidades:** μV² (integral de PSD sobre la banda)

#### Banda Alpha (8-12 Hz)
- **Canal:** Pz (parietal)
- **Rango:** 8.0 a 12.0 Hz
- **Cálculo:**
  ```python
  idx_band = np.logical_and(freqs >= 8.0, freqs <= 12.0)
  alpha_power = np.trapz(psd[idx_band], freqs[idx_band])
  ```
- **Unidades:** μV² (integral de PSD sobre la banda)

**Método de integración:** Regla del trapecio (`np.trapz`) para calcular el área bajo la curva PSD en cada banda.

---

## 7. Cálculo del Índice de Carga Cognitiva

### 7.1 Fórmula del Ratio

**Archivo:** `signal_worker.py` - Líneas 207-209

```python
ratio = theta_power / alpha_power
```

**Fórmula matemática:**

\[
\text{Cognitive Load Ratio} = \frac{\text{Theta Power}_{Fz}}{\text{Alpha Power}_{Pz}}
\]

### 7.2 Interpretación

**Valores del ratio:**
- **Ratio > 1:** Mayor potencia Theta relativa → **Mayor carga cognitiva**
- **Ratio < 1:** Mayor potencia Alpha relativa → **Menor carga cognitiva**
- **Ratio ≈ 1:** Carga cognitiva moderada

**Justificación neurofisiológica:**
- **Theta (4-7 Hz) en Fz:** Asociado con esfuerzo mental, atención sostenida, memoria de trabajo
- **Alpha (8-12 Hz) en Pz:** Asociado con relajación, procesamiento pasivo, estado de reposo
- **Ratio Theta/Alpha:** Indicador robusto de carga cognitiva en tareas de atención

### 7.3 Validación

**Archivo:** `signal_worker.py` - Línea 207

```python
if alpha_power > 0:
    ratio = theta_power / alpha_power
    return ratio, theta_power, alpha_power
```

- Verifica que `alpha_power > 0` para evitar división por cero
- Retorna `None` si no hay suficientes datos o alpha_power es cero

---

## 8. Actualización en Tiempo Real

### 8.1 Timer de Cálculo

**Archivo:** `main.py` - Líneas 68-70

```python
self.ratio_timer = QTimer()
self.ratio_timer.timeout.connect(self.calculate_and_update_ratio)
self.ratio_timer.start(1000)  # Cada 1 segundo
```

**Frecuencia de actualización:**
- **Cálculo del ratio:** Cada 1 segundo
- **Ventana de análisis:** 2 segundos (500 muestras)
- **Solapamiento:** 50% entre ventanas consecutivas

### 8.2 Flujo de Actualización

**Archivo:** `main.py` - Método `calculate_and_update_ratio()`

1. **Timer dispara** cada 1 segundo
2. **Llama a** `signal_worker.get_cognitive_load_ratio()`
3. **Obtiene** ventana de 2 segundos del buffer circular
4. **Calcula** bandpower Theta y Alpha
5. **Calcula** ratio = Theta / Alpha
6. **Emite** resultado a la UI para visualización

### 8.3 Visualización del Ratio

**Archivo:** `ui_main.py` - Método `update_ratio_plot()`

- **Gráfico:** Línea temporal del ratio
- **Actualización:** Cada vez que se calcula un nuevo ratio
- **Buffer:** Últimos 300 puntos (~5 minutos a 1 Hz)
- **Eje X:** Tiempo relativo (segundos desde el presente)
- **Eje Y:** Valor del ratio (adimensional)

---

## 9. Resumen del Flujo Completo

```
┌─────────────────────────────────────────────────────────────┐
│ 1. ADQUISICIÓN LSL                                           │
│    AURA → LSL Stream → pull_sample()                         │
│    Frecuencia: 250 Hz                                        │
│    Formato: 8 canales en nanovolts                           │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. PREPROCESAMIENTO                                          │
│    • Conversión a NumPy array                                 │
│    • Filtro Notch 60 Hz                                       │
│    • Filtro Pasabanda 1-40 Hz                                │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. ALMACENAMIENTO                                            │
│    • RingBuffer circular (1000 muestras = 4 seg)            │
│    • Mantiene estado de filtros                              │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ├─────────────────┐
                       ▼                 ▼
        ┌──────────────────────┐  ┌──────────────────────┐
        │ 4. VISUALIZACIÓN     │  │ 5. ANÁLISIS          │
        │    Raw signals       │  │    Espectral         │
        │    (8 canales)       │  │    • Welch PSD       │
        └──────────────────────┘  │    • Bandpower       │
                                  └──────────┬───────────┘
                                             │
                                             ▼
                                  ┌──────────────────────┐
                                  │ 6. CÁLCULO RATIO     │
                                  │    Theta_Fz /        │
                                  │    Alpha_Pz           │
                                  └──────────┬───────────┘
                                             │
                                             ▼
                                  ┌──────────────────────┐
                                  │ 7. VISUALIZACIÓN    │
                                  │    Ratio temporal    │
                                  │    (actualización    │
                                  │     cada 1 segundo)  │
                                  └──────────────────────┘
```

---

## 10. Parámetros Técnicos Clave

| Parámetro | Valor | Descripción |
|-----------|-------|-------------|
| **Tasa de muestreo** | 250 Hz | Muestras por segundo del dispositivo |
| **Número de canales** | 8 | Canales EEG simultáneos |
| **Filtro Notch** | 60 Hz, Q=30 | Eliminación de ruido de línea |
| **Filtro Pasabanda** | 1-40 Hz, orden 4 | Rango de frecuencias EEG |
| **Ventana de análisis** | 2 segundos (500 muestras) | Para cálculo de bandpower |
| **Banda Theta** | 4-7 Hz | Canal Fz (frontal) |
| **Banda Alpha** | 8-12 Hz | Canal Pz (parietal) |
| **Frecuencia de cálculo** | 1 Hz | Cada segundo se actualiza el ratio |
| **Método espectral** | Welch's periodogram | Estimación de PSD |
| **Ventana Welch** | 250 muestras (1 segundo) | Tamaño de segmento |
| **Solapamiento Welch** | 50% (125 muestras) | Entre segmentos |

---

## 11. Consideraciones de Rendimiento

### 11.1 Optimizaciones Implementadas

1. **Buffer de emisión:** Reduce señales de 250 Hz a ~25 Hz
2. **Submuestreo en UI:** Actualiza gráficos cada 2 muestras
3. **Submuestreo en logging:** Guarda cada 5 muestras (50 Hz)
4. **Timers ajustados:** Gráficos a 10 FPS, ratio a 1 Hz

### 11.2 Uso de Memoria

- **RingBuffer:** ~32 KB (1000 muestras × 8 canales × 4 bytes)
- **Buffers de gráficos:** ~40 KB por gráfico
- **Logging:** Depende de duración del experimento (submuestreado)

---

## 12. Referencias Técnicas

- **LSL (Lab Streaming Layer):** Protocolo de streaming de datos biomédicos
- **Welch's Method:** Welch, P. D. (1967). "The use of fast Fourier transform for the estimation of power spectra"
- **Filtros Digitales:** Oppenheim & Schafer, "Discrete-Time Signal Processing"
- **Bandas EEG:** Klimesch, W. (1999). "EEG alpha and theta oscillations reflect cognitive and memory performance"

---

**Última actualización:** Diciembre 2025 
**Versión del sistema:** 1.0

