# Resumen del pipeline AI-Cognitive (EEG, carga cognitiva)

Documento de síntesis: dataset, preprocesado, CSVs, features, modelos, resultados (3 vs 2 clases) y conclusiones sobre qué ha funcionado mejor.

---

## 1. Descripción del dataset

### 1.1 Origen

- **Datos crudos**: archivos `.txt` en  
  `raw_data/Arithmetic_Data/` y `raw_data/Stroop_Data/`.
- **Tareas**: *Arithmetic* y *Stroop*.
- **Condiciones por nombre de archivo** (prefijo antes del `-` y número de sujeto):
  - `natural` — línea base
  - `lowlevel`, `midlevel`, `highlevel` — distintos niveles de carga en la tarea
- **Sujetos**: identificador numérico en el nombre del archivo (p. ej. `natural-1.txt` → sujeto 1).
- **Estructura de cada línea**: valores numéricos por canal (EEG u otras señales), opcionalmente timestamp al final; a veces prefijo tipo `L123:`.

### 1.2 Escala aproximada (referencia de ejecuciones típicas)

| Nivel | Cantidad orientativa |
|-------|----------------------|
| Filas en `eeg_all_samples.csv` | ~2,77 M |
| Canales numéricos (`v0`…`v23`) | 24 |
| Archivos `.txt` por tarea | 4 prefijos × 15 sujetos ≈ 60 por carpeta de tarea |

### 1.3 Etiquetas definidas en el proyecto

| Columna | Significado |
|---------|-------------|
| `condition_4` | 4 niveles del paper: normal, low, mid, high (según prefijo) |
| `load_3` | 3 clases: low (lowlevel), normal (natural), high (mid+high) |
| `load_2` | **Binario**: `normal` (solo natural) vs `alta` (lowlevel + midlevel + highlevel) |

---

## 2. Preprocesado

### 2.1 Conversión raw → CSV (`eeg_convert_raw_to_clean.py`)

- Lectura línea a línea, limpieza de prefijos, separación por comas.
- Último campo con formato de fecha/hora tratado como **timestamp** (no como canal).
- Valores no numéricos o vacíos → `0.0`.
- **Relleno** de filas con menos columnas hasta el máximo de canales visto.
- **Sin filtrado ni normalización** en este paso: se conserva la escala del dispositivo/grabación.

### 2.2 Preprocesado antes del bandpower (`eeg_make_window_features.py`)

Por **ventana de 4 s** (500×4 muestras a 250 Hz) y **por canal**, **antes** de estadísticos en frecuencia y buena parte de los de tiempo:

1. **Band-pass Butterworth** 1–40 Hz, orden 4, fase cero (`sosfiltfilt`).
2. **Detrend lineal** en la ventana (quita DC y deriva lenta).
3. **Notch** a 50 Hz y 60 Hz (fase cero, Q≈30).

Las estadísticas en **tiempo** (media, std, min, max, RMS, valor absoluto medio) y el **bandpower** se calculan sobre esta señal ya filtrada.

### 2.3 Escalado en modelos tabulares (`eeg_train_window_classifier.py`)

- **StandardScaler** ajustado **solo en train**, aplicado a train y test (sin fuga).
- Valores **NaN/Inf** → 0 y **clip** a ±1e10 para estabilidad.

---

## 3. Creación de los CSV

| Archivo | Script | Contenido |
|---------|--------|-----------|
| `csv/eeg_all_samples.csv` | `eeg_convert_raw_to_clean.py` | Una fila por instante de muestreo: `v0…vN`, timestamp, `subject_id`, `task_type`, `condition_4`, `load_3`, **`load_2`**, `file_name`. |
| `csv/eeg_window_features.csv` | `eeg_make_window_features.py` | Una fila por **ventana** (4 s, solapamiento 50 %): cientos de features por canal + metadatos (`load_2`, etc.). |

**Orden recomendado**: convertir raw → (opcional) `eeg_analyze_binary_dataset.py` → features → entrenamiento.

---

## 4. Features por ventana

- **Ventana**: 4 s, paso 2 s (50 % solapamiento), **Fs = 250 Hz**.
- **Por canal** (tras el preprocesado descrito):
  - Tiempo: mean, std, min, max, RMS, abs_mean.
  - Frecuencia: potencias en bandas **delta, theta, alpha, beta, gamma** (Welch), potencia total, **potencias relativas** por banda, ratios theta/alpha y beta/alpha, **entropía espectral** (1–40 Hz).
- Número de columnas de features: del orden de **~480** numéricas además de metadatos.

---

## 5. Protocolo de entrenamiento (común)

- **Split por `file_name`** (estratificado por la etiqueta objetivo), no por ventana, para **evitar fuga** entre ventanas del mismo registro.
- **SMOTE** solo en **train** (balanceo de clases en el espacio de features).
- **Pesos de clase** según tarea:
  - **3 clases (`load_3`)**: refuerzo a la clase minoritaria *low* donde aplica.
  - **2 clases (`load_2`)**: peso mayor a **normal** para mejorar su recall (p. ej. factor 3.5 frente a *alta*), ajustable con `LOAD2_NORMAL_CLASS_WEIGHT`.

---

## 6. Modelos entrenados

### 6.1 Sobre features de ventana (`eeg_train_window_classifier.py`)

| Modelo | Notas |
|--------|--------|
| **RandomForest** | 300 árboles, `class_weight` según tarea. |
| **HistGradientBoosting** | En binario, pesos por índice de clase compatibles con el codificador interno. |
| **SVM RBF** | C=5, `gamma='scale'`. |
| **LogisticRegression** | Binario sin `multi_class` explícito. |
| **MLP** | Capas (128, 64); en versiones antiguas de sklearn **sin** `sample_weight` en `fit`. |

### 6.2 Sobre señal cruda ventaneada (`eeg_train_cnn_lstm.py`)

- Entrada: tensores **(canales × tiempo)** por ventana de 4 s, normalización por canal en el subset.
- Arquitectura: **CNN 1D** (dos bloques conv+pool) → **LSTM bidireccional** → capas densas.
- Entrenamiento con **muestreo ponderado** y **CrossEntropy** con pesos por clase; dispositivo **CUDA / MPS / CPU** según disponibilidad.

### 6.3 Análisis exploratorio

- **`eeg_analyze_binary_dataset.py`**: conteos por `load_2`, por tarea, archivos, crosstab con `condition_4`, estimación de número de ventanas.

---

## 7. Resultados obtenidos (referencia de las corridas del proyecto)

Los valores exactos pueden variar ligeramente según split y semilla; aquí se resumen **órdenes de magnitud y patrones** observados.

### 7.1 Tres clases (`load_3`: low, normal, high)

| Aspecto | Observación |
|---------|-------------|
| **Accuracy global** | Baja (~6 % en 4 clases `condition_4`; ~50–56 % en `load_3` si el modelo predomina *normal*). |
| **Clase `low`** | **Recall muy bajo o 0**: el EEG de *lowlevel* se confunde con *normal* o *high* en el espacio de features usado. |
| **Clase `normal`** | A menudo sobre-representada en las predicciones. |
| **Conclusión** | Separar **tres niveles de carga** con ventanas cortas y estas features es **muy difícil**; el problema es más de **solapamiento de señal** que del algoritmo concreto. |

### 7.2 Dos clases (`load_2`: normal vs alta)

| Modelo (orientativo) | Accuracy | Recall **normal** | Recall **alta** |
|----------------------|----------|-------------------|-----------------|
| **RandomForest** | ~78–80 % | **~40 %** | ~91 % |
| HistGradientBoosting | ~80 % | ~30 % | ~97 % |
| SVM RBF | ~77 % | ~30 % | ~93 % |
| LogisticRegression | ~68 % | ~37 % | ~79 % |
| MLP | ~77 % | ~26 % | ~94 % |

**Dataset binario (filas):** ~31 % *normal*, ~69 % *alta*; a nivel archivo, 15 *normal* vs 45 *alta*.

### 7.3 CNN + LSTM (principalmente con etiquetas de 3 clases en pruebas tempranas)

- Con **3 clases**, el modelo tendía a **colapsar** a una sola clase (p. ej. todo *normal*), con accuracy ~24 % y sin predicciones útiles para *high*/*low*.
- Tras **pesos de clase** y muestreo balanceado, la pérdida bajaba poco; la tarea **binaria** en la CNN es más razonable y alineada con lo que funcionó en tabular.

---

## 8. Qué ha funcionado mejor y por qué

### 8.1 Mejor configuración práctica hasta ahora

1. **Problema binario `load_2` (normal vs alta)**  
   - Reduce la ambigüedad: *alta* agrupa cualquier bloque con tarea; *normal* es solo *natural*.  
   - Es **más coherente con la separabilidad** del EEG en ventanas de pocos segundos (línea base vs “con tarea”).

2. **Modelo: Random Forest sobre features de ventana**  
   - Mejor **recall de *normal*** (~40 %) frente a boosting, SVM, MLP y CNN-LSTM en las pruebas realizadas.  
   - Los árboles **combinan bien** bandas tipo **alpha relativo, beta, ratios** con `class_weight` para no ignorar la clase minoritaria.  
   - **HistGB** y **SVM** priorizan mucho **alta** (alto recall en *alta*, bajo en *normal*) porque la frontera en el espacio de features sigue empujando la mayoría de ventanas hacia *alta*.

3. **Preprocesado 1–40 Hz + detrend + notch**  
   - Hace las features espectrales **más interpretables** y estables (menos DC y línea eléctrica dominando).

4. **Split por archivo**  
   - Evita optimismo artificial por correlación entre ventanas del mismo `.txt`.

### 8.2 Por qué 3 clases y CNN-LSTM fueron peores en la práctica

- **Tres clases**: *low* queda **entre** *normal* y *high* en el espacio de features; los modelos lineales/de árbol y la CNN no encuentran un cluster claro para *low*.  
- **CNN-LSTM**: muchos parámetros, dataset moderado en número de **archivos**; sin afinar mucho hiperparámetros y con desbalance, el mínimo de la pérdida sigue siendo **predecir la clase mayoritaria**. Sobre **features tabulares** el RF ya resolvía mejor el binario.

### 8.3 Líneas de mejora posibles

- Subir `LOAD2_NORMAL_CLASS_WEIGHT` si se prioriza **aún más** detectar *normal* (intercambiando precisión en *alta*).  
- Validación **leave-one-subject-out** para medir generalización entre personas.  
- Más datos o ventanas más largas si el protocolo lo permite.

---

## 9. English summary

- **Dataset**: Multi-channel EEG-like streams from Arithmetic/Stroop tasks; labels from filename prefixes.  
- **Pipeline**: Raw → `eeg_all_samples.csv` (+ `load_2`) → window features (4 s, band-pass 1–40 Hz, detrend, notch, spectral + time features) → `eeg_window_features.csv`.  
- **3-class (`load_3`)**: Poor, especially **zero recall on *low*** — classes overlap heavily in feature space.  
- **2-class (`load_2`)**: **Random Forest** works best (~**80%** accuracy, ~**40%** *normal* recall, ~**91%** *alta* recall); boosting/SVM/MLP favor *alta* at the expense of *normal*.  
- **CNN-LSTM**: Struggled on 3-class; binary + more tuning would be the sensible next step.  
- **Why RF wins**: Handles mixed feature scales after scaling, strong with band-power features, and **class_weight** effectively protects the minority *normal* class.

---

*Documento generado como memoria técnica del proyecto AI-Cognitive. Ajustar números concreto si se cambian semillas, pesos o conjuntos de test.*
