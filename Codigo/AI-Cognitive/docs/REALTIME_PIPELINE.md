# Real-time cognitive-load classification — pipeline and results

Documento técnico de la iteración 2026-06-25 sobre el dataset Arithmetic + Stroop
(`raw_data/`). Foco: **clasificación en tiempo real** de carga cognitiva con
calibración por sujeto y arquitectura jerárquica.

## 1. Dataset

- 15 sujetos × 4 condiciones (`natural`, `lowlevel`, `midlevel`, `highlevel`)
- 24 canales (`v0…v23`), Fs = 250 Hz
- ~5 369 ventanas de 4 s con solape del 50 % tras preprocesado
- Targets:
  - `load_2` — binario: `normal` (natural) vs `alta` (low+mid+high)
  - `load_3` — ternario: `normal` / `low` / `high` (mid se funde con high)

## 2. Preprocesado por ventana

Aplicado por canal antes de cualquier feature espectral
([`eeg_make_window_features.py`](../scripts/eeg_make_window_features.py)):

1. Band-pass Butterworth 1–40 Hz (orden 4, fase cero).
2. Detrend lineal por ventana.
3. Notch a 50 Hz y 60 Hz (Q ≈ 30).

## 3. Conjunto de features (656 columnas)

Por cada uno de los 24 canales:

| Bloque | Features |
|---|---|
| Tiempo | mean, std, min, max, rms, abs_mean |
| Band-power Welch | δ, θ, α, β, γ, total, relativas |
| Ratios | θ/α, β/α, θ/β, engagement = β/(α+θ), load = θ/α |
| Hjorth | activity, mobility, complexity |
| Entropía | spectral entropy en 1–40 Hz |

Más **asimetrías log-α y log-β** por pares de canales opuestos (proxy de FAA
sin montaje conocido).

## 4. Protocolo de evaluación

- **Split por sujeto** — Leave-One-Subject-Out (LOSO).
- **Calibración por sujeto**: z-score de todas las features de cada sujeto
  contra la media/std de **sus propias ventanas `normal`** (línea base).
  Esto emula la fase real-time de 30 s de baseline al inicio de cada sesión.
- **Streaming honesto** — el sujeto held-out reserva el 50 % de sus ventanas
  baseline para calibrar; sólo el resto va a evaluación.

## 5. Modelo

Arquitectura **jerárquica en dos etapas** (RandomForest 400 árboles,
`class_weight="balanced"`):

1. **Stage 1**: `normal` vs `alta` (load_2).
2. **Stage 2**: dentro de `alta`, `low` vs `high`.
3. Probabilidades conjuntas → predicción final entre `{normal, low, high}`.
4. **EMA** sobre las probabilidades de salida (α = 0.4) para suavizar
   la traza temporal.

## 6. Resultados

### 6.1 Tabla comparativa (RandomForest)

| Setting | `load_2` acc | recall normal | macro-f1 | `load_3` acc | recall low |
|---|---|---|---|---|---|
| Split por archivo (baseline doc) | 0.78 | 0.40 | 0.67 | 0.39 | 0.00 |
| LOSO sin calibrar | 0.77 | 0.46 | 0.70 | 0.61 | 0.00 |
| **LOSO + calibración por sujeto** | **0.85** | **0.74** | **0.82** | 0.65 | 0.00 |
| Streaming honesto (50 % baseline reservada, jerárquico + EMA) | **0.81 ± 0.14** | — | **0.76 ± 0.13** | 0.63 ± 0.10 | 0.00 |

Detalle del barrido por sujeto en
[`scripts/eeg_streaming_sweep.py`](../scripts/eeg_streaming_sweep.py).

### 6.2 Dispersión inter-sujeto (binario, streaming honesto)

| Grupo | Sujetos | Rango acc |
|---|---|---|
| Calibración funciona bien | 4, 6, 8, 11, 12, 14 | 0.92–0.97 |
| Aceptable | 1, 7, 9, 13, 15 | 0.74–0.84 |
| Marginal / fallida | 2, 3, 5, 10 | 0.54–0.70 |

Los sujetos del último grupo presentan **drift dentro de la sesión**:
la baseline temprana ya no representa bien las ventanas posteriores.

### 6.3 Limitación estructural en 3 clases

Con cualquier modelo (RF, HistGB, SVM, MLP, CNN-LSTM), la clase `low` tiene
**recall ≈ 0**: en el espacio de features actuales se confunde casi por
completo con `high`. No es un defecto del clasificador — es solapamiento
real entre las ventanas `lowlevel` y `mid/highlevel` que la transformación
tiempo→frecuencia + Hjorth no resuelve.

### 6.4 Latencia

- Cómputo de features por ventana (4 s, 24 canales, 656 features): ~30 ms en CPU.
- Inferencia jerárquica RF: **1.6 ms/ventana**.
- EMA: < 0.01 ms.
- Predicción cada 2 s — margen amplísimo.

## 7. Scripts

| Script | Función |
|---|---|
| [`eeg_convert_raw_to_clean.py`](../scripts/eeg_convert_raw_to_clean.py) | Raw .txt → `csv/eeg_all_samples.csv` con etiquetas |
| [`eeg_make_window_features.py`](../scripts/eeg_make_window_features.py) | Ventanas + preprocesado + 656 features → `csv/eeg_window_features.csv` |
| [`eeg_train_window_classifier.py`](../scripts/eeg_train_window_classifier.py) | Entrena RF/HistGB/SVM/LogReg/MLP con split por archivo |
| [`eeg_loso_eval.py`](../scripts/eeg_loso_eval.py) | LOSO sin calibración |
| [`eeg_loso_calibrated.py`](../scripts/eeg_loso_calibrated.py) | LOSO + calibración por sujeto |
| [`eeg_realtime_infer.py`](../scripts/eeg_realtime_infer.py) | Prototipo streaming (jerárquico + EMA) |
| [`eeg_streaming_sweep.py`](../scripts/eeg_streaming_sweep.py) | Barrido honesto sobre los 15 sujetos |

## 8. Próximos pasos

1. **Recalibración online**: actualizar la baseline cada N minutos con
   ventanas predichas como `normal` con alta confianza.
2. **Reemplazar RF por LightGBM** para reducir tamaño del modelo en producción.
3. Para reconquistar la clase `low`:
   - Añadir **sample entropy** y **permutation entropy**.
   - **Coherencia entre canales** (band-coherence en θ y α).
   - Plantear como **regresión ordinal** (puntaje 0-3) en vez de clasificación dura.
4. **Recolectar más sujetos** o más sesiones por sujeto para reducir la
   varianza intersujeto observada.

## 9. ¿Es esto material de paper?

Sí, con framing honesto. Ver
[`PAPER_FEASIBILITY.md`](PAPER_FEASIBILITY.md) para el análisis.
