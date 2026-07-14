# Data-Experimento-Jeronimo: segmentación por eventos

## Origen

Script: `analysis/pipeline/step_jeronimo_segment_by_events.py`

- **Carpetas que empiezan con `data`** = baseline → se lee `baseline_raw_*.csv`, label unificado a `baseline`.
- **Resto** = tarea → se lee `AURA_RAW___*.csv`, se segmenta por la columna **Event** (inicio de cada etapa).

## Formato de salida

Cada CSV tiene:

- `timestamp`: segundos (relativos al inicio de la sesión en tareas; absolutos en baseline).
- `label`: `baseline` (baseline) o `pre`, `segment_4`, `segment_5`, `segment_6`, etc. (tareas).
- `channel_0` … `channel_7`: Fp1, Fp2, F3, Fz, F4, P3, Pz, P4.

## Heterogeneidad de eventos

Los **códigos de evento** (4, 5, 6 / 7, 8 / 1, 2, 3 / 21, 22, 23 / 10, etc.) **no son iguales entre sesiones**. Dependen del protocolo de cada grabación. Las etiquetas son `segment_N` donde N es el valor del evento; puedes mapearlas después (p. ej. segment_4 → low_load, segment_5 → high_load) según tu protocolo.

Algunas tareas no tienen eventos (Event siempre 0); en esos casos todo el archivo queda con label `pre`.

## Uso posterior

Estos CSVs tienen el mismo esquema de canales que el experimento Rafa (`timestamp`, `label`, `channel_0`…`channel_7`). Puedes usarlos como entrada a un pipeline de análisis (p. ej. cognitive load) una vez definas la correspondencia entre `segment_N` y tus condiciones.
