# Análisis del preprocesado antes del bandpower

## Resumen: qué se hace ahora

**Antes del bandpower no se aplica ningún preprocesado de señal.**  
La ventana de cada canal se pasa directamente a `scipy.signal.welch()`.

---

## 1. Paso 1: `eeg_convert_raw_to_clean.py` (raw → CSV)

| Acción | ¿Se hace? | Detalle |
|--------|-----------|--------|
| Leer líneas del .txt | Sí | Por archivo |
| Quitar prefijo `Lxxx:` | Sí | Solo limpieza de formato |
| Separar por comas | Sí | Campos numéricos + timestamp al final |
| Convertir a float | Sí | Valores no numéricos o vacíos → `0.0` |
| Rellenar filas cortas | Sí | Con ceros para igualar número de columnas |
| **Filtrado (band-pass, notch)** | **No** | — |
| **Detrend / eliminar DC** | **No** | — |
| **Re-referenciado** | **No** | — |
| **Detección/eliminación de artefactos** | **No** | — |
| **Normalización por canal/sujeto** | **No** | — |

Salida: `csv/eeg_all_samples.csv` con columnas `v0..v23` (y metadatos). Los valores son los mismos que en el raw, solo reordenados y con etiquetas.

---

## 2. Paso 2: `eeg_make_window_features.py` (ventanas → features)

Flujo hasta el bandpower:

1. Cargar `eeg_all_samples.csv` y quitar filas con `condition_4 == "unknown"`.
2. Agrupar por `(file_name, subject_id, task_type, condition_4, load_3)`.
3. Para cada grupo, tomar ventanas de **2 s** (500 muestras a 250 Hz), step 1 s (50 % solapamiento).
4. Para cada ventana y cada canal:
   - **Dominio tiempo**: se calculan `mean`, `std`, `min`, `max` sobre el segmento **sin ningún preprocesado**.
   - **Dominio frecuencia**: se llama a `bandpower(segmento_canal, FS=250, band)`.

Dentro de `bandpower()`:

```python
freqs, psd = welch(data, sf, nperseg=nperseg)
# luego integración en la banda (trapezoid)
```

Es decir: el segmento de ese canal se pasa **tal cual** a `welch()`; no hay:

- Filtrado (band-pass 1–40 Hz, notch 50/60 Hz).
- Detrend (eliminar tendencia/DC).
- Re-referenciado.
- Normalización.

---

## 3. Consecuencias

- **DC y deriva**: Influyen en la PSD y en bandpowers (sobre todo en bandas bajas).
- **Ruido de red (50/60 Hz)**: Entra en beta/gamma y puede dominar la potencia.
- **Aliasing / altas freqs**: Por encima de 40 Hz no hay filtrado antialiasing; si hay ruido de alta frecuencia, contamina.
- **Escalas distintas entre canales**: Algunos canales tienen rango 0–255, otros miles; el bandpower refleja esa escala sin normalizar.

Por eso es importante **añadir preprocesado antes del bandpower** (al menos band-pass y detrend) si quieres bandpowers más limpios y comparables.

---

## 4. Preprocesado implementado (antes del bandpower)

En `eeg_make_window_features.py` se aplica **por ventana y por canal**, en este orden:

1. **Band-pass** Butterworth 1–40 Hz, orden 4, fase cero (`sosfiltfilt`).
2. **Detrend** lineal en la ventana (elimina DC y deriva).
3. **Notch** 50 Hz y 60 Hz (fase cero, Q=30) para atenuar ruido de red.

Las estadísticas en tiempo (mean, std, min, max) y el bandpower se calculan sobre esta señal preprocesada. Constantes en el script: `BANDPASS_LOW_HZ`, `BANDPASS_HIGH_HZ`, `BANDPASS_ORDER`, `NOTCH_FREQS_HZ`, `NOTCH_QUALITY`.
