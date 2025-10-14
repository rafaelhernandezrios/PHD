# AURA_Software — EEG LSL Toolkit (Doctoral Research)

Resumen (ES): Este repositorio contiene un conjunto de scripts en Python para adquirir señales EEG desde el software Aura mediante Lab Streaming Layer (LSL), aplicar filtrado (notch y pasabanda), filtrado Kalman, calcular potencia espectral por bandas (bandpower/PSD), visualizar en tiempo real y guardar resultados en CSV. Está diseñado como evidencia y soporte metodológico para un proyecto de doctorado que requiere trazabilidad de la conexión con EEG, preprocesamiento, métricas de potencia por banda y almacenamiento de datos.

Abstract (EN): This repository provides Python scripts to acquire EEG from the Aura software via Lab Streaming Layer (LSL), apply notch and band‑pass filtering, perform Kalman filtering, compute spectral band power (bandpower/PSD), visualize signals in real time, and persist results to CSV. It serves as documented evidence of EEG connectivity, preprocessing, feature computation, and data export for a doctoral research project.

---

### Contents / Contenido

- `1_LSL_read_raw_data.py`: Lee el stream LSL crudo (`AURA`) y muestra muestras en consola. Reads raw LSL EEG (`AURA`) and prints samples.
- `2_LSL_filter_raw_data.py`: Filtra (notch + banda 1–50 Hz) y publica `AURAFilteredEEG`; además aplica Kalman y publica `AURAKalmanFilteredEEG`. Filters and publishes both filtered and Kalman‑filtered streams.
- `3_EEG_Bandpower.py`: Recibe `AURAFilteredEEG` (Aura) o un stream EEG y publica un stream de 5×N canales con bandpower normalizado (`EEG_BANDPOWER_X`). Computes normalized bandpower.
- `4_LSL_3channel_Bandpower.py`: Lanza el filtrado y calcula PSD por bandas para 3 canales, publicando `AURAPSD`. Starts filtering and computes 3‑channel PSD bands.
- `5_LSL_Graphic.py`: Visualización en tiempo real de `AURAFilteredEEG` y `AURAKalmanFilteredEEG` con Tkinter/Matplotlib. Real‑time plotting.
- `csvsaver.py`: Suscribe a `AURAPSD` y guarda CSV de PSD por canal/banda. Saves PSD to CSV.
- `csvsaverRAW.py`: Suscribe a `AURAFilteredEEG` y `AURAKalmanFilteredEEG` y guarda CSV con timestamp. Saves filtered/Kalman to CSV.
- `runall.py`: Orquesta el flujo de 3‑canales (PSD + guardado). Orchestrates 3‑channel PSD workflow.
- `USB-AURA/`: Variante de scripts para flujos USB/ejemplos. USB variant scripts.

---

### EEG/LSL Architecture / Arquitectura EEG/LSL

- Aura software emite un stream LSL crudo llamado `AURA` (activar opción LSL en la UI de Aura).
- `2_LSL_filter_raw_data.py`:
  - Entrada: `AURA` (o `AURA_Power` si el emisor usa ese nombre; ajústalo en el código si es necesario).
  - Salidas: `AURAFilteredEEG` (señal filtrada) y `AURAKalmanFilteredEEG` (señal filtrada + Kalman).
- `3_EEG_Bandpower.py`:
  - Entrada Aura: `AURAFilteredEEG`.
  - Salida: `EEG_Headset` con tipo `EEG_BANDPOWER_X`, dimensión 5×N + 1 (timestamp). Orden por bandas: alpha, beta, gamma, delta, theta por canal.
- `4_LSL_3channel_Bandpower.py`:
  - Entrada: `AURAFilteredEEG`.
  - Salida: `AURAPSD` con 5×3 valores (Delta/Theta/Alpha/Beta/Gamma por cada uno de 3 canales).

Nota (ES): Si tu versión de Aura publica `AURA_Power` o un nombre distinto, actualiza la línea de `resolve_stream('name', ...)` en los scripts para coincidir.
Note (EN): If your Aura version publishes `AURA_Power` or a different name, update `resolve_stream('name', ...)` accordingly.

---

### Requirements / Requisitos

- Python 3.9+ (Windows tested)
- `pylsl`, `numpy`, `scipy`, `matplotlib`, `filterpy`
- Tkinter (incluido en Python para Windows), `tk`/GUI available

Install (EN):

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install --upgrade pip
pip install pylsl numpy scipy matplotlib filterpy
```

Instalar (ES):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install pylsl numpy scipy matplotlib filterpy
```

---

### Quickstart Workflows / Flujos Rápidos

1) Minimal raw read (EN/ES):

```powershell
python 1_LSL_read_raw_data.py
```

2) Filtering + Kalman + real‑time visualization:

```powershell
start python 2_LSL_filter_raw_data.py
python 5_LSL_Graphic.py
```

3) 3‑channel PSD + CSV saving:

```powershell
python runall.py
```

4) 8‑channel normalized bandpower stream:

```powershell
python 2_LSL_filter_raw_data.py
python 3_EEG_Bandpower.py
```

Ensure Aura is running and LSL streaming is enabled (`AURA`).
Asegúrate de que Aura esté ejecutándose y que LSL esté activo (`AURA`).

---

### Scripts — Details / Detalles

- `1_LSL_read_raw_data.py`:
  - ES: Resuelve `AURA` y muestra `(timestamp, sample)` en consola.
  - EN: Resolves `AURA` and prints `(timestamp, sample)`.

- `2_LSL_filter_raw_data.py`:
  - ES: Aplica notch (50 Hz) y pasabanda (1–50 Hz) por canal; publica `AURAFilteredEEG`. Aplica Kalman multicanal y publica `AURAKalmanFilteredEEG`. Frecuencia de muestreo por defecto: 250 Hz. Número de canales por defecto: 3 (ajustable).
  - EN: Applies notch (50 Hz) and band‑pass (1–50 Hz), publishes `AURAFilteredEEG`, then multi‑channel Kalman to `AURAKalmanFilteredEEG`. Default fs=250 Hz; default channels=3.

- `3_EEG_Bandpower.py`:
  - ES: Lee `AURAFilteredEEG` (modo Aura) o cualquier `type==EEG` LSL. Calcula potencias por banda (alpha, beta, gamma, delta, theta) por canal usando FFT y normaliza a [0,1]. Publica `EEG_BANDPOWER_X` con 5×N valores + timestamp.
  - EN: Reads `AURAFilteredEEG` (Aura) or generic EEG. Computes per‑band power via FFT, normalizes to [0,1], publishes `EEG_BANDPOWER_X` (5×N + timestamp).

- `4_LSL_3channel_Bandpower.py`:
  - ES: Lanza el filtrado, toma 3 canales, calcula PSD (Welch) por bandas, publica `AURAPSD` (5×3). También imprime valores enviados.
  - EN: Starts filtering, computes Welch PSD per band over 3 channels and publishes `AURAPSD` (5×3). Prints sent values.

- `5_LSL_Graphic.py`:
  - ES: UI Tkinter con dos gráficos actualizados cada 100 ms para `AURAFilteredEEG` y `AURAKalmanFilteredEEG`.
  - EN: Tkinter UI with two plots, updates every 100 ms for both streams.

- `csvsaver.py`:
  - ES: Suscribe a `AURAPSD` (type `PSD`) y guarda columnas `[Delta, Theta, Alpha, Beta, Gamma] × 3` en `datos_psd2.csv`.
  - EN: Subscribes to `AURAPSD` and writes `[Delta, Theta, Alpha, Beta, Gamma] × 3` to `datos_psd2.csv`.

- `csvsaverRAW.py`:
  - ES: Suscribe a `AURAFilteredEEG` y `AURAKalmanFilteredEEG` y guarda dos CSV con timestamp en el nombre; columnas: `Timestamp, Canal1..3`.
  - EN: Subscribes to both filtered streams and writes two timestamped CSVs; columns: `Timestamp, Channel1..3`.

- `runall.py`:
  - ES: Inicia `4_LSL_3channel_Bandpower.py`, luego `csvsaver.py` y `csvsaverRAW.py` en ventanas separadas.
  - EN: Starts the 3‑channel PSD producer and both CSV savers.

---

### Data Formats / Formatos de Datos

- `AURAFilteredEEG` / `AURAKalmanFilteredEEG`: float32, N canales, fs=250 Hz por defecto, unidades en µV tras `SCALE_FACTOR_EEG`.
- `EEG_BANDPOWER_X`: 5×N valores normalizados + timestamp (orden: alpha→beta→gamma→delta→theta por canal).
- `AURAPSD`: 5×3 valores (Delta/Theta/Alpha/Beta/Gamma por cada canal).
- CSV de `csvsaver.py`: 15 columnas (5 bandas × 3 canales).
- CSV de `csvsaverRAW.py`: 4 columnas por archivo (`Timestamp + 3 canales`).

---

### Reproducibility & Logging / Reproducibilidad y Registro

- Guarda versiones de scripts y tiempos de ejecución en commits de Git para trazabilidad. Keep versioned commits for traceability.
- Los CSV llevan marca temporal; conserva metadatos de condiciones experimentales (participante, tarea, sesión). Keep experimental metadata alongside CSVs.

---

### Troubleshooting / Solución de Problemas

- No se encuentra el stream `AURA`:
  - ES: Verifica que Aura tenga habilitado LSL y que el nombre del stream coincida. Aumenta `time.sleep()` antes de resolver el stream.
  - EN: Ensure Aura has LSL enabled and names match; increase `time.sleep()` before resolving streams.

- Permisos de Tkinter/GUI:
  - ES: Ejecuta en entorno con soporte de ventanas (no WSL sin X‑server). Cierra otras UIs que bloqueen el hilo.
  - EN: Run on a desktop session; avoid headless environments without display.

- Dependencias faltantes:
  - `pip install pylsl numpy scipy matplotlib filterpy`

- Nombre de stream distinto (`AURA_Power`):
  - ES/EN: Edita `resolve_stream('name', '...')` en los scripts para empatar el nombre real.

---

### How to Cite / Cómo Citar

ES: Si este repositorio contribuye a tu trabajo, por favor cita como “AURA_Software — EEG LSL Toolkit (Año, versión de commit)”. Puedes complementar con cita a LSL:

> Kothe, C. A., & Makeig, S. (2013). BCILAB: a platform for brain-computer interface development. Frontiers in Neuroinformatics.

EN: If this repository contributes to your work, please cite “AURA_Software — EEG LSL Toolkit (Year, commit version)”. Also cite LSL as appropriate (see above).

---

### License / Licencia

Pending — Research/educational use. Update with your preferred license (e.g., MIT).

---

### Acknowledgements / Agradecimientos

- Lab Streaming Layer (LSL)
- Aura hardware/software team

# Aura: Manual de Usuario

¡Bienvenido a Aura! Nuestro sistema de biosensado más reciente, desarrollado con una atención meticulosa a la experiencia del usuario, te ofrece una manera intuitiva y potente de interactuar con los datos de tu EEG cap.

## Descripción General

Aura consta de una placa de biosensado y un gorro de EEG diseñados bajo una misma dirección de diseño. Este sistema proporciona una experiencia mejorada al usuario a través de la impresión en pantalla en el gorro, que indica la posición de cada electrodo, y una carcasa de biosensado más resistente y delgada impresa en 3D en SLS (sinterización láser selectiva).

## Tutorial de Uso

### Conexión en Vivo

Este será tu panel principal. Desde aquí, puedes iniciar una conexión en vivo haciendo clic en el botón "Conexión".

<img src=https://github.com/edgarhernandez94/ANUIES/blob/main/WAVEX/AURA_SDK/AuraTutorial1/AuraTutorial_1.png  width="60%">

1. **Selecciona el Puerto COM Identificado**: Aquí selecciona tu puerto COM identificado para Aura y haz clic en "Conectar". En pocos segundos, comenzarás a recibir datos en vivo.

<img src=https://github.com/edgarhernandez94/ANUIES/blob/main/WAVEX/AURA_SDK/AuraTutorial1/AuraTutorial_2.png width="60%">

### Configuración de Electrodos

Asegúrate de que tus electrodos estén bien conectados y posicionados.

1. **Configuración de Electrodos**: Haz clic en "Archivo" -> "Configurar Electrodos" para abrir el panel de configuración. También puedes hacer clic en el icono de Configuración rápida en la esquina superior izquierda del cerebro.

<img src=https://github.com/edgarhernandez94/ANUIES/blob/main/WAVEX/AURA_SDK/AuraTutorial1/AuraTutorial_3.png width="60%">

2. **Ajuste de los Números de Canal de Electrodo**: Mueve los números de canal de electrodo para reflejar la configuración real de tu auricular Aura. Luego, guarda la configuración de posiciones con el botón sobre el cuero cabelludo.

<img src=https://github.com/edgarhernandez94/ANUIES/blob/main/WAVEX/AURA_SDK/AuraTutorial1/AuraTutorial_4.png width="60%">

3. **Indicadores de Calidad de Contacto en Tiempo Real**: Durante el ajuste de los electrodos en el cuero cabelludo, los indicadores de calidad de contacto se irán volviendo "buenos". El color rojo indica una mala conexión entre el cuero cabelludo y el electrodo correspondiente del número de canal del dispositivo, por lo que puede requerirse más gel conductor.

<img src=https://github.com/edgarhernandez94/ANUIES/blob/main/WAVEX/AURA_SDK/AuraTutorial1/AuraTutorial_5.png width="60%">

### Grabación

1. **Inicio de la Grabación**: Simplemente haz clic en el botón "Grabar" para comenzar a grabar.

<img src=https://github.com/edgarhernandez94/ANUIES/blob/main/WAVEX/AURA_SDK/AuraTutorial1/AuraTutorial_7.png width="60%">

2. **Directorio de Archivos Generados**: Durante una conexión en vivo, los archivos se generan automáticamente en la carpeta de Documentos de tu computadora, bajo ...Usuarios\[ tú ]\Documentos\Aura EEG Records. Puedes elegir tu propia carpeta de destino yendo a "Archivo" -> "Configuración" o haciendo clic en el icono de Guardar arriba del botón de Grabación.

<img src=https://github.com/edgarhernandez94/ANUIES/blob/main/WAVEX/AURA_SDK/AuraTutorial1/AuraTutorial_8.png width="60%">

3. **Registro de Eventos**: Durante la grabación, puedes presionar la barra espaciadora de tu teclado para registrar un evento en tu archivo CSV para referencia futura. Se indicará con un cambio de color en la luz de Evento.

 <img src=https://github.com/edgarhernandez94/ANUIES/blob/main/WAVEX/AURA_SDK/AuraTutorial1/AuraTutorial_9.png width="60%">

### Transmisión

Mientras una conexión en vivo está activa, puedes activar o desactivar la casilla de verificación "LSL Stream Out". De esta manera, enviarás tus señales en bruto a través de la red. También funciona con tus sesiones grabadas.

<img src=https://github.com/edgarhernandez94/ANUIES/blob/main/WAVEX/AURA_SDK/AuraTutorial1/AuraTutorial_11.png width="60%">

#### Uso Avanzado: Integración con LSL y Python

Este documento explica los pasos para obtener datos en bruto del software Aura mediante el protocolo Lab Stream Layer (LSL) utilizando el lenguaje de programación Python. Para obtener más información sobre LSL, consulta: [Lab Stream Layer Documentation](https://labstreaminglayer.readthedocs.io/).

**Pasos:**

1. Abre el software Aura y comienza a leer las señales de EEG.

2. Haz clic en la casilla de verificación "LSL" ubicada encima de los gráficos para iniciar la transmisión de los datos de EEG en bruto. El software Aura genera automáticamente una capa de transmisión LSL llamada "AURA".

3. Para visualizar los datos de EEG en bruto, ejecuta el siguiente archivo: `1_LSL_read_raw_data.py`.

4. Para filtrar los datos en bruto provenientes de la transmisión LSL 'AURA', ejecuta el siguiente archivo: `2_LSL_filter_raw_data.py`. Este script leerá los datos en bruto, los filtrará y generará una nueva transmisión llamada "AURAFilteredEEG" y "AURAKalmanFilteredEEG".

5. Para utilizar solo 3 canales y calcular la potencia espectral de la frecuencia (PSD), ejecuta el archivo `4_LSL_3channel_Banpower`. Este script tomará solo los primeros tres electrodos y calculará la PSD para obtener la EEG_Bandpower, creando una nueva transmisión llamada "AURAPSD".

6. Para visualizar "AURAFilteredEEG" y "AURAKalmanFilteredEEG", puedes ejecutar `5_LSL_Graphic`.

7. Para calcular la potencia de banda de los datos filtrados, ejecuta el siguiente archivo:

   Este archivo recibe una transmisión LSL de ocho canales EEG con datos de EEG filtrados llamada "AURAFilteredEEG". La salida es una transmisión de 40 canales llamada 'EEG_BANDPOWER_X' que contiene valores normalizados de potencia de banda de las señales de entrada en la siguiente forma:

   - Delta CH1
   - Delta CH2
   - ...
   - Delta CH8

   - Theta CH1~CH8
   - Alpha CH1~CH8
   - Beta CH1~CH8
   - Gamma CH1~CH8

### Reproducción

1. **Carga de un Archivo CSV Grabado Anteriormente**: Haz clic en "Archivo" -> "Cargar Conjunto de Datos". Por ejemplo, puedes cargar una muestra de demostración haciendo clic en "Siguiente".

<img src=https://github.com/edgarhernandez94/ANUIES/blob/main/WAVEX/AURA_SDK/AuraTutorial1/AuraTutorial_12.png width="60%">

2. **Exploración de Funciones de Reproducción**: Juega con todas las escalas y filtros para las señales. Por ejemplo, prueba el control deslizante de suavidad en el panel FFT.

<img src=https://github.com/edgarhernandez94/ANUIES/blob/main/WAVEX/AURA_SDK/AuraTutorial1/AuraTutorial_13.png width="60%">

3. **Visualización de Datos en el Modelo 3D del Cerebro**: Puedes rotar el modelo 3D del cerebro y seleccionar qué banda deseas visualizar con los botones debajo del cuero cabelludo.

<img src=https://github.com/edgarhernandez94/ANUIES/blob/main/WAVEX/AURA_SDK/AuraTutorial1/AuraTutorial_14.png width="60%">

### Acelerómetro y Giroscopio

También puedes verificar las señales de acelerómetro y giroscopio de tu Aura haciendo clic en esta pestaña.

<img src=https://github.com/edgarhernandez94/ANUIES/blob/main/WAVEX/AURA_SDK/AuraTutorial1/AuraTutorial_15.png width="60%">

Esto te desplegará el siguiente panel de visualización.

<img src=https://github.com/edgarhernandez94/ANUIES/blob/main/WAVEX/AURA_SDK/AuraTutorial1/AuraTutorial_16.png width="60%">

---

¡Esperamos que este manual te sea útil para sacar el máximo provecho de Aura! Si tienes alguna pregunta o sugerencia, no dudes en ponerte en contacto con nuestro equipo de soporte. ¡Disfruta explorando las capacidades de Aura integrado en Wavex!
