# Protocolo de sesión v2 — plataforma Cognitive-load

Actualizado el 18 ago 2026, tras la revisión del pipeline de adquisición.
Este documento cubre **qué cambió**, **qué revisar antes de cada participante** y
**qué archivos salen** de cada sesión.

---

## 1. Qué cambió respecto a la versión anterior

| Antes | Ahora |
|---|---|
| Sólo se guardaba la señal filtrada | Se guardan **crudo y filtrado**, ambos a 250 Hz |
| Decimación 250→50 Hz tirando 4 de cada 5 muestras, sin anti-alias | Sin decimación en adquisición; si hace falta 50 Hz se genera offline con filtro anti-alias |
| Todo en RAM, se escribía una vez al final | **Escritura incremental** a disco (flush cada ~250 muestras o 2 s) |
| Los ensayos del Stroop no se guardaban en ningún lado | `trials_stroop.csv` + marcas en `events.csv` |
| Sin marcas de evento; granularidad = bloque | Marcas de onset y respuesta **en reloj LSL** |
| `performance.now()` y reloj LSL sin relación | Handshake de sincronía; incertidumbre reportada en la UI |
| 5–11 % de timestamps hacia atrás | `proc_clocksync \| proc_dejitter \| proc_monotonize` → 0 % |
| Transitorio del filtro escrito dentro del CSV | Estados iniciales escalados por la primera muestra |
| Sin detección de pérdida de muestras ni de electrodo muerto | Monitor de salud en la barra lateral, 1 Hz, sobre la señal **cruda** |
| Gráfica de control filtrada a 7–13 Hz | Banda ancha 0.5–45 Hz (se ven deriva, pops y EMG) |
| 24 llamadas a `lfilter` por muestra | `pull_chunk` + filtrado vectorizado (3 llamadas por bloque) |
| Stroop: palabra y tinta al azar independientes (~25 % congruente, sin registrar) | 120 ensayos balanceados: **50 % congruentes**, 30 por tinta, sin más de 2 tintas iguales seguidas |
| Respuesta con mouse | Respuesta con teclado **D / F / J / K** |
| Sin fijación, ISI fijo de 300 ms | Cruz de fijación con ISI jitterado 600–1000 ms |

---

## 2. Checklist antes de cada participante

1. **Lanza** con `run_gui.command`.
2. **Connect AURA.** Mira la barra lateral, sección *Acquisition health*:
   - `rate` debe decir **~250 / 250 Hz** en verde. Si dice otra cosa, para.
   - `lost` debe estar en **0.0 %**.
   - Los ocho chips (Fp1…P4) deben estar en **verde**. Un chip rojo = electrodo
     muerto o saturado; recolócalo antes de seguir.
3. **Setup Session** con el ID del participante. Aquí empieza la grabación y se
   crea la carpeta de sesión.
4. Corre Baseline → Low Load (lectura) → High Load (Stroop).
   - Al pulsar *Start Stroop*, el botón dice "Syncing clock…" durante ~1 s.
     Cuando `clock sync` en la barra lateral muestre un valor **± unos pocos ms**,
     la tarea arranca. Si dice `failed`, no grabes ERPs: el marcador de estímulo
     no será fiable.
5. **Save & Finish** al terminar. Los datos ya están en disco; el botón sólo
   fuerza el último flush y te dice dónde quedaron.

> Vigila la barra lateral **durante** la sesión, no sólo al principio. Un
> electrodo se puede soltar a mitad del Stroop y el chip se pondrá rojo.

---

## 3. Qué sale de cada sesión

```
data/data_<participante>/<participante>_<AAAAMMDD_HHMMSS>/
├── eeg_raw.csv         250 Hz · cuentas de ADC sin tocar
├── eeg_filtered.csv    250 Hz · notch 60 Hz + pasabanda 1-40 Hz causal
├── events.csv          marcas en reloj LSL
├── trials_stroop.csv   conducta por ensayo
└── session.json        metadatos: Fs, montaje, filtros, offset de reloj
```

### `eeg_raw.csv` / `eeg_filtered.csv`

`timestamp_lsl, phase, label, ecological_modality, channel_0 … channel_7`

**Analiza siempre desde `eeg_raw.csv`.** El filtrado es causal y de una pasada;
para análisis offline conviene refiltrar con `filtfilt` (fase cero). El archivo
filtrado está para inspección rápida y para que la UI muestre algo, no para ser
la fuente del análisis.

### `events.csv`

`timestamp_lsl, event, phase, detail`

Eventos: `session_start`, `phase_change`, `clock_sync`, `stroop_block_start`,
`stroop_onset`, `stroop_response`, `stroop_block_end`, `stream_lost`,
`low_sampling_rate`, `session_end`.

**Esta es la fuente autoritativa para epochar.** La columna `phase` de los
archivos de señal se asigna por bloque de adquisición y puede desviarse hasta
~100 ms en los bordes; `events.csv` lleva el instante exacto.

### `trials_stroop.csv`

`trial_index, block, word, ink, congruent, response, correct, rt_ms,
onset_lsl, response_lsl, onset_perf_ms, response_perf_ms, clock_offset, clock_rtt_ms`

`response` vale `none` si el participante no contestó en la ventana de 2000 ms;
en ese caso `rt_ms` queda vacío.

---

## 4. Cómo epochar

```python
import pandas as pd, numpy as np, json

raw = pd.read_csv("eeg_raw.csv")
ev  = pd.read_csv("events.csv")

onsets = ev[ev.event == "stroop_onset"].copy()
onsets["congruent"] = onsets.detail.apply(lambda d: json.loads(d)["congruent"])

fs = 250.0
t  = raw.timestamp_lsl.to_numpy()
x  = raw[[f"channel_{i}" for i in range(8)]].to_numpy()
# ...aquí: filtfilt, re-referencia, rechazo de artefactos...

pre, post = 0.2, 0.8   # segundos
epochs = []
for _, r in onsets.iterrows():
    i0 = np.searchsorted(t, r.timestamp_lsl - pre)
    i1 = i0 + int((pre + post) * fs)
    if i1 <= len(x):
        epochs.append(x[i0:i1])
epochs = np.asarray(epochs)      # (n_trials, n_samples, n_channels)
```

---

## 5. Compatibilidad con los scripts de `analysis/`

Los scripts existentes esperan el formato antiguo (`eeg_data_*.csv` con columna
`timestamp`). Para generarlo desde una sesión v2:

```bash
python analysis/utils/convert_session_to_legacy.py \
    data/data_<participante>/<participante>_<AAAAMMDD_HHMMSS>/
```

Produce `eeg_data_<sesion>.csv` a 50 Hz, pero con dos diferencias respecto al
recorder viejo, ambas intencionales:

- decima con `scipy.signal.decimate` (**filtro anti-alias**), en vez de tirar
  muestras a pelo;
- refiltra el crudo con `filtfilt` (**fase cero**);
- los canales muertos salen como `NaN` en lugar de como una traza casi plana que
  parece limpia.

Con `--keep-250` emite el archivo sin decimar.

---

## 6. Lo que sigue pendiente

- **Latencia de pantalla.** El onset se marca tras el segundo `requestAnimationFrame`,
  o sea cuando la palabra ya está pintada. Queda la latencia física del panel
  (~10–25 ms, bastante constante). Si vas a publicar latencias de N200, mídela
  una vez con un fotodiodo y réstala como constante.
- **Verificar el mapeo de canales** `0=Fp1 … 3=Fz … 6=Pz` contra la configuración
  real de AURA. Está codificado por comentario y ya hubo un commit corrigiendo un
  mapeo (`49e3786b`).
- **Ventana de Welch de 2 s** para el índice θ/α: da resolución de 1 Hz y sólo 3
  segmentos, así que el valor en vivo es ruidoso. Subir a 4 s si el índice
  importa durante la sesión (para el análisis offline es irrelevante).
- **Limpiar `venv/` vs `.venv/`**: conviven dos entornos y `run_gui.command` usa
  `venv`. Borra el que no uses.
- `requirements.txt` no incluye `PyWavelets`, que `core/waaf_filter.py` necesita.
