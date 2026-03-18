# AI-Cognitive — EEG stress/load classification

Pipeline para clasificar condición de estrés/carga cognitiva a partir de señales EEG (ventanas de 2 s, solapamiento 50%).

## Estructura del proyecto

```
AI-Cognitive/
├── csv/
│   ├── eeg_all_samples.csv    # Salida del paso 1 (muestras limpias)
│   └── eeg_window_features.csv # Salida del paso 2 (features por ventana)
├── raw_data/                  # Datos crudos (Arithmetic_Data, Stroop_Data)
├── scripts/
│   ├── eeg_convert_raw_to_clean.py   # 1) Raw .txt → CSV con etiquetas
│   ├── eeg_make_window_features.py   # 2) CSV muestras → features por ventana
│   └── eeg_train_window_classifier.py # 3) Entrenar y evaluar modelos
└── requirements.txt
```

## Cómo ejecutar (orden)

Para ejecutar **todo el pipeline** de una vez (desde la raíz `AI-Cognitive/`):

```bash
python scripts/eeg_convert_raw_to_clean.py && \
python scripts/eeg_make_window_features.py && \
python scripts/eeg_train_window_classifier.py
```

O paso a paso:

1. **Convertir raw → CSV limpio**  
   Coloca los `.txt` en `raw_data/Arithmetic_Data` y `raw_data/Stroop_Data`.  
   Luego:
   ```bash
   python scripts/eeg_convert_raw_to_clean.py
   ```
   Genera `csv/eeg_all_samples.csv`.

2. **Calcular features por ventana**  
   ```bash
   python scripts/eeg_make_window_features.py
   ```
   Genera `csv/eeg_window_features.csv` (ventanas 2 s, step 1 s; bandas delta/theta/alpha/beta/gamma + estadísticos en tiempo).

3. **Entrenar y evaluar modelos**  
   ```bash
   python scripts/eeg_train_window_classifier.py
   ```
   Entrena RandomForest e HistGradientBoosting para **3 clases**: `low`, `normal`, `high` (target `load_3`).  
   El split es **por archivo** (no por ventana) para evitar fugas; se usa **StandardScaler** (fit solo en train), **SMOTE** y **class_weight** reforzado para la clase `low`.

   **Mapeo de etiquetas** (en `eeg_convert_raw_to_clean.py`):
   - **low** = lowlevel (carga baja)
   - **normal** = natural (línea base)
   - **high** = midlevel + highlevel (carga media/alta)

## Por qué "low" puede tener recall 0

Con el mapeo actual, la clase **low** (lowlevel) suele confundirse con **high** en el espacio de features: el EEG de carga baja puede ser muy parecido al de carga media/alta en ventanas de 2 s, y el modelo aprende sobre todo a separar **normal** (natural) del resto. No es un error del pipeline; es una limitación del poder discriminativo de estas features para "low". Para mejorarlo: más features (p. ej. asimetrías, filtrado band-pass), ventanas más largas o validación por sujeto.

## Si los resultados son pobres

- **Escalado**: Ya se aplica `StandardScaler`; evita que canales con escalas muy distintas dominen.
- **Valores extremos**: Se hace clip a ±1e10 y reemplazo de inf/nan para evitar explosión en band-power ratios.
- **Desbalance**: Se usa `class_weight="balanced"`. Si "low" sigue con recall 0, prueba:
  - **SMOTE** u oversampling de minoritarias.
  - **Métricas**: f1 macro y matriz de confusión por clase, no solo accuracy.
- **Pocos archivos/sujetos**: Con ~60 archivos, el test puede ser poco estable. Considera validación cruzada por **sujeto** (leave-one-subject-out) para estimar mejor la generalización.
- **Features**: Las importancias suelen estar en medias/máx/mín por canal. Prueba:
  - Reducir ruido: **filtrado band-pass** antes de band-power (p. ej. 1–40 Hz).
  - Más features EEG: asimetrías (izq/der), entropía, coherencia entre canales.
- **Preprocesado raw**: Asegura que los `.txt` son voltaje o escala consistente; si algún canal está en unidades distintas (p. ej. 0–255 vs µV), normaliza por canal antes de `eeg_make_window_features.py`.

## Dependencias

- Python 3.9+
- numpy, pandas, scipy, scikit-learn (ver `requirements.txt`)
