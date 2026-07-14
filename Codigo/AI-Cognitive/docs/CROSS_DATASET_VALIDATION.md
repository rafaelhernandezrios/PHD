# Validación cruzada en otros datasets

Iteración 2026-06-25.

## Por qué

El paper EMBC necesita un argumento contra "N=15 es muy pequeño". Replicar
la pipeline en otros datasets EEG de carga cognitiva permite:

1. **Cuantificar generalización** del protocolo (calibración + jerárquico + EMA).
2. **Subir N efectivo total** (15 + 36 + 48 = 99 sujetos).
3. **Aislar el efecto del headset/Fs/canales**: si el método aguanta en hardware
   y paradigmas distintos, el mensaje metodológico se sostiene.

## Datasets

| Dataset | Sujetos | Canales | Fs | Paradigma | Estado |
|---|---|---|---|---|---|
| **Arithmetic/Stroop** (propio) | 15 | 24 (`v0…v23`) | 250 Hz | Stroop + arithmetic, 4 niveles | ✅ |
| **Zyma 2019** (PhysioNet `eegmat`) | 36 | 19 EEG nombrados | 500 Hz → 250 Hz resample | Subtracción mental serial, 2 niveles | ✅ |
| **STEW** (Lim 2018, IEEE Dataport) | 48 | 14 (Emotiv EPOC) | 128 Hz | SIMKAP multitasking, 2 niveles | ⏳ Pendiente descarga manual |

## Mismo protocolo en los tres

- Ventanas de 4 s con solape 50 %.
- Preprocesado: band-pass 1–40 Hz + detrend + notch 50/60 Hz.
- Features: time-domain, band-power Welch (δθαβγ), ratios, Hjorth, entropía
  espectral.
- LOSO con calibración por sujeto (z-score contra propias ventanas
  `normal`).
- 50 % de la baseline del sujeto held-out reservada para calibrar; el
  resto va a evaluación honesta.
- RandomForest 400 árboles, `class_weight="balanced"`.

## Resultados binarios (normal vs alta)

| Dataset | acc | macro-f1 | recall normal | recall alta | Notas |
|---|---|---|---|---|---|
| **Arithmetic/Stroop** | **0.81 ± 0.14** | 0.76 ± 0.13 | — (binary aggregated) | — | jerárquico + EMA |
| **Zyma** | **0.76 ± 0.14** | 0.69 ± 0.20 | 0.96 ± 0.05 | 0.47 ± 0.34 | binario directo (no jerárquico) |
| **STEW** | — | — | — | — | pendiente descarga |

**Interpretación**: la pipeline calibrada por sujeto rinde de forma comparable
(diferencia 5 pts de accuracy) en un dataset con:
- Headset distinto (Neurovisor/clínico vs equipo del primer dataset)
- 19 canales en vez de 24
- Frecuencia de muestreo distinta (500 Hz vs 250 Hz)
- Tarea distinta (subtracción mental vs Stroop/arithmetic)
- Sesiones muy cortas (1 min de tarea, 3 min de baseline)

## Hallazgos en Zyma

- `recall normal` muy alto (96 %) y `recall alta` bajo (47 %) — el modelo
  se sesga hacia la clase mayoritaria. Esto es consecuencia del desbalance
  3:1 (3186 normal vs 1080 alta) por la corta duración de la tarea.
- Sujetos con `recall alta = 0` (subj 04, 12) son aquellos con ventanas
  baseline que ya contienen actividad mental similar a la tarea — la
  calibración con la baseline propia no separa lo suficiente.
- Los sujetos donde funciona muy bien (06, 17, 20, 28, 30, 33: acc ≥ 0.94)
  son los que tienen mayor "puntuación de calidad" en `subject-info.csv`.

## Scripts

| Script | Función |
|---|---|
| [`zyma_make_features.py`](../scripts/zyma_make_features.py) | EDF → window features con canales nombrados |
| [`zyma_streaming_sweep.py`](../scripts/zyma_streaming_sweep.py) | LOSO + calibración + RF sobre Zyma |

## Pendiente

- [ ] Descargar STEW manualmente desde IEEE Dataport.
- [ ] Crear `stew_make_features.py` (texto plano, 14 canales Emotiv, 128 Hz).
- [ ] Reportar tabla final con los tres datasets para el paper.

## Por qué no zero-shot transfer

El plan original era "entrenar en AS, testear en Zyma". No es viable directamente
porque las 656 features del modelo AS están indexadas a 24 canales `v0…v23`
sin correspondencia montaje-a-montaje con los 19 canales nombrados de Zyma.

Alternativas para el paper:
1. **Reportar como replicación independiente** (lo hecho ahora) — válido
   metodológicamente, es el patrón estándar en literatura EEG cross-dataset.
2. **Channel-agnostic features** (media/std/min/max de cada feature a través
   de todos los canales del sujeto) — habilitaría transfer real, pero pierde
   información espacial. Future work.
3. **Mapeo manual canales AS → posiciones 10-20** si conseguimos la
   documentación del headset usado en AS.
