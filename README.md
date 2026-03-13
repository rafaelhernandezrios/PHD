# Proyecto de Doctorado - Procesamiento de Señales EEG y Análisis de Carga Cognitiva / PhD Project - EEG Signal Processing and Cognitive Load Analysis

Repositorio completo de código desarrollado para investigación doctoral en neurotecnología, enfocado en el procesamiento de señales EEG y análisis de carga cognitiva utilizando el dispositivo AURA.

Complete code repository developed for doctoral research in neurotechnology, focused on EEG signal processing and cognitive load analysis using the AURA device.

---

## 📋 Descripción General / General Description

Este proyecto contiene herramientas y plataformas completas para:

This project contains complete tools and platforms for:

- **Adquisición en tiempo real** de señales EEG mediante Lab Streaming Layer (LSL)
- **Procesamiento y limpieza** de señales EEG con múltiples técnicas de filtrado
- **Análisis de carga cognitiva** mediante índices espectrales (Theta/Alpha ratio)
- **Experimentación estructurada** con protocolos controlados y tareas cognitivas
- **Visualización científica** en tiempo real de señales y métricas

- **Real-time acquisition** of EEG signals via Lab Streaming Layer (LSL)
- **Processing and cleaning** of EEG signals with multiple filtering techniques
- **Cognitive load analysis** through spectral indices (Theta/Alpha ratio)
- **Structured experimentation** with controlled protocols and cognitive tasks
- **Scientific visualization** in real-time of signals and metrics

---

## 🗂️ Estructura del Proyecto / Project Structure

```
PHD/
├── Cognitive-load/          # Plataforma principal de experimentación
│   ├── main.py             # Aplicación principal con interfaz gráfica
│   ├── signal_worker.py    # Worker thread para adquisición LSL
│   ├── experiment_logic.py # Máquina de estados del protocolo experimental
│   ├── ui_main.py          # Interfaz gráfica PyQt5
│   ├── requirements.txt    # Dependencias del módulo
│   ├── README.md           # Documentación específica del módulo
│   ├── FLUJO_CARGA_COGNITIVA.md  # Documentación técnica detallada
│   └── data_*/             # Datos experimentales por usuario
│
├── Clean EEG/              # Scripts de procesamiento y limpieza de señales
│   ├── plot_fp1.py        # Pipeline completo de procesamiento EEG
│   ├── read_csv_channel1.py  # Lectura y análisis de datos CSV
│   ├── USB-AURA/          # Scripts de ejemplo para dispositivo AURA
│   │   ├── 1_LSL_read_raw_data.py
│   │   ├── 2_LSL_filter_raw_data.py
│   │   ├── 3_LSL_plot_data.py
│   │   ├── 4_LSL_plot_data_three_channels.py
│   │   ├── 5_LSL_realtime_filter_plot.py
│   │   └── 6_LSL_realtime_plot_simple.py
│   └── requirements.txt   # Dependencias del módulo
│
├── Codigos Ejemplo/        # Códigos de ejemplo y utilidades
│   ├── 1_LSL_read_raw_data.py      # Lectura básica de datos LSL
│   ├── 2_LSL_filter_raw_data.py    # Filtrado de señales
│   ├── 3_EEG_Bandpower.py          # Cálculo de bandpower
│   ├── 4_LSL_3channel_Bandpower.py # Bandpower multi-canal
│   ├── 5_LSL_Graphic.py            # Visualización gráfica
│   ├── csvsaver.py                 # Guardado de datos filtrados
│   ├── csvsaverRAW.py              # Guardado de datos raw
│   └── runall.py                   # Script para ejecutar múltiples procesos
│
├── AI-Cognitive/           # Pipelines de clasificación y análisis offline
│   ├── raw_data/           # Datos EEG crudos organizados por tarea
│   ├── csv/                # Features y muestras EEG agregadas (local-only, archivos muy grandes ignorados por Git)
│   └── scripts/            # Scripts para limpiar, extraer ventanas y entrenar clasificadores
│
└── README.md               # Este archivo
```

---

## 🚀 Inicio Rápido / Quick Start

### Requisitos del Sistema / System Requirements

- **Python 3.9+** (recomendado 3.9 o superior)
- **Dispositivo AURA** con drivers instalados
- **Lab Streaming Layer (LSL)** configurado
- **Windows/Linux/Mac** (probado principalmente en Windows)

### Instalación / Installation

1. **Clonar el repositorio / Clone the repository**
```bash
git clone https://github.com/tu-usuario/PHD.git
cd PHD
```

2. **Crear entorno virtual / Create virtual environment**

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Linux/Mac:**
```bash
python3 -m venv venv
source venv/bin/activate
```

3. **Instalar dependencias / Install dependencies**

Para el módulo de Cognitive-load:
```bash
cd Cognitive-load
pip install -r requirements.txt
```

Para el módulo de Clean EEG:
```bash
cd Clean EEG
pip install -r requirements.txt
```

---

## 📦 Módulos Principales / Main Modules

### 1. Cognitive-load - Plataforma de Experimentación

**Descripción / Description:**
Plataforma completa para experimentación EEG en tiempo real con análisis de carga cognitiva. Sistema integrado de adquisición, procesamiento, visualización y logging de datos.

Complete platform for real-time EEG experimentation with cognitive load analysis. Integrated system for acquisition, processing, visualization and data logging.

**Características principales / Main features:**
- ✅ Adquisición en tiempo real vía LSL (8 canales, 250 Hz)
- ✅ Filtrado digital: Notch 60 Hz + Pasabanda 1-40 Hz
- ✅ Análisis espectral en tiempo real (Método de Welch)
- ✅ Cálculo de bandpower: Theta (4-7 Hz) y Alpha (8-12 Hz)
- ✅ Índice de carga cognitiva: Ratio Theta_Fz / Alpha_Pz
- ✅ Protocolo experimental estructurado (Baseline, Baja/Alta carga)
- ✅ Tareas cognitivas integradas (Stroop, lectura pasiva)
- ✅ Interfaz gráfica científica con PyQt5
- ✅ Data logging organizado por usuario

**Uso / Usage:**
```bash
cd Cognitive-load
python main.py
```

**Documentación completa:** Ver [Cognitive-load/README.md](Cognitive-load/README.md)

---

### 2. Clean EEG - Procesamiento y Limpieza de Señales

**Descripción / Description:**
Scripts para procesamiento avanzado de señales EEG con múltiples etapas de filtrado y técnicas de limpieza de artefactos.

Scripts for advanced EEG signal processing with multiple filtering stages and artifact cleaning techniques.

**Pipeline de procesamiento / Processing pipeline:**
1. **Señal Original**: Datos sin procesar del canal Fp1
2. **Filtro Bandpass**: 0.2-50 Hz
3. **Common Average Reference (CAR)**: Re-referenciación
4. **Filtro Stopband**: Eliminación de 60 Hz (ruido de red eléctrica)
5. **WAAF**: Wavelet-Assisted Adaptive Filter para remover artefactos oculares

**Uso / Usage:**
```bash
cd "Clean EEG"
python plot_fp1.py
```

**Scripts disponibles / Available scripts:**
- `plot_fp1.py`: Pipeline completo de procesamiento con visualización
- `read_csv_channel1.py`: Lectura y análisis de archivos CSV
- `USB-AURA/`: Scripts de ejemplo para trabajar con dispositivo AURA

---

### 3. Codigos Ejemplo - Ejemplos y Utilidades

**Descripción / Description:**
Códigos de ejemplo para aprender y trabajar con señales EEG, LSL y análisis básico. Incluye scripts modulares y reutilizables.

Example codes for learning and working with EEG signals, LSL and basic analysis. Includes modular and reusable scripts.

**Scripts incluidos / Included scripts:**
- `1_LSL_read_raw_data.py`: Lectura básica de datos desde LSL
- `2_LSL_filter_raw_data.py`: Aplicación de filtros digitales
- `3_EEG_Bandpower.py`: Cálculo de potencia espectral en bandas
- `4_LSL_3channel_Bandpower.py`: Análisis multi-canal
- `5_LSL_Graphic.py`: Visualización gráfica de señales
- `csvsaver.py`: Guardado de datos filtrados a CSV
- `csvsaverRAW.py`: Guardado de datos raw a CSV
- `runall.py`: Script para ejecutar múltiples procesos simultáneamente

**Uso / Usage:**
```bash
cd "Codigos Ejemplo"
python 1_LSL_read_raw_data.py
```

---

## 🔧 Configuración Técnica / Technical Configuration

### Parámetros del Sistema / System Parameters

| Parámetro / Parameter | Valor / Value | Descripción / Description |
|----------------------|---------------|--------------------------|
| **Tasa de muestreo** / Sampling rate | 250 Hz | Muestras por segundo del dispositivo |
| **Canales EEG** / EEG channels | 8 | Canales simultáneos |
| **Filtro Notch** / Notch filter | 60 Hz, Q=30 | Eliminación de ruido de línea |
| **Filtro Pasabanda** / Bandpass filter | 1-40 Hz, orden 4 | Rango de frecuencias EEG |
| **Ventana de análisis** / Analysis window | 2 segundos (500 muestras) | Para cálculo de bandpower |
| **Banda Theta** / Theta band | 4-7 Hz | Canal Fz (frontal) |
| **Banda Alpha** / Alpha band | 8-12 Hz | Canal Pz (parietal) |
| **Frecuencia de cálculo** / Calculation frequency | 1 Hz | Actualización del ratio cada segundo |

### Mapeo de Canales / Channel Mapping

- **Canal 0**: Fz (Frontal) - Usado para análisis Theta
- **Canal 1-3**: Canales adicionales
- **Canal 4**: Pz (Parietal) - Usado para análisis Alpha
- **Canal 5-7**: Canales adicionales

---

## 📊 Formato de Datos / Data Format

Los archivos CSV guardados contienen las siguientes columnas:

The saved CSV files contain the following columns:

- `timestamp`: Timestamp de la muestra / Sample timestamp
- `phase`: Fase del experimento (setup, baseline_eyes_open, etc.)
- `label`: Etiqueta descriptiva de la fase
- `channel_0` a `channel_7`: Valores de los 8 canales EEG (filtrados)

---

## 🛠️ Tecnologías Utilizadas / Technologies Used

- **Python 3.9+** - Lenguaje de programación principal
- **PyQt5** - Interfaz gráfica de usuario
- **pyqtgraph** - Visualización científica en tiempo real
- **NumPy** - Procesamiento numérico y arrays
- **SciPy** - Filtros digitales y análisis espectral
- **Pandas** - Manejo de datos y exportación CSV
- **pylsl** - Comunicación con Lab Streaming Layer
- **matplotlib** - Visualización de gráficos estáticos
- **PyWavelets** - Transformadas wavelet para limpieza de artefactos

---

## 📚 Documentación Adicional / Additional Documentation

- **[Cognitive-load/README.md](Cognitive-load/README.md)** - Documentación completa de la plataforma de experimentación
- **[Cognitive-load/FLUJO_CARGA_COGNITIVA.md](Cognitive-load/FLUJO_CARGA_COGNITIVA.md)** - Documentación técnica detallada del flujo de procesamiento

---

## 🎯 Casos de Uso / Use Cases

### Experimentación de Carga Cognitiva
1. Conectar dispositivo AURA vía LSL
2. Ejecutar protocolo experimental estructurado
3. Realizar tareas cognitivas (Stroop, lectura)
4. Analizar ratio Theta/Alpha en tiempo real
5. Guardar datos para análisis posterior

### Procesamiento Offline de Señales
1. Cargar datos EEG desde CSV
2. Aplicar pipeline completo de filtrado
3. Remover artefactos oculares con WAAF
4. Visualizar señales procesadas
5. Exportar resultados

### Desarrollo y Prototipado
1. Usar scripts de ejemplo para aprender LSL
2. Modificar parámetros de filtrado
3. Crear nuevos análisis espectrales
4. Integrar con otros sistemas

---

## 📝 Notas de Desarrollo / Development Notes

- Los datos experimentales se guardan en carpetas `data_*` que no deben versionarse
- Cada módulo tiene su propio `requirements.txt` para independencia de dependencias
- El código está optimizado para tiempo real con buffers circulares y submuestreo inteligente
- La documentación técnica detallada está en formato Markdown dentro de cada módulo
- Algunos archivos EEG agregados (por ejemplo, CSV con **todas** las muestras y tamaño > 100 MB) se mantienen solo de forma local y se excluyen explícitamente con `.gitignore` para cumplir con las restricciones de GitHub.

- Experimental data is saved in `data_*` folders that should not be versioned
- Each module has its own `requirements.txt` for dependency independence
- Code is optimized for real-time with circular buffers and intelligent subsampling
- Detailed technical documentation is in Markdown format within each module
- Some aggregated EEG files (e.g., CSV files with **all** samples and size > 100 MB) are kept locally only and are explicitly excluded via `.gitignore` to comply with GitHub file size limits.

---

## 🤝 Contribuciones / Contributions

Este es un proyecto de investigación doctoral. Para contribuciones o colaboraciones, por favor contactar al autor.

This is a doctoral research project. For contributions or collaborations, please contact the author.

---

## 📧 Contacto / Contact

Para preguntas, sugerencias o colaboraciones relacionadas con este proyecto de investigación:

For questions, suggestions or collaborations related to this research project:

- Abrir un issue en el repositorio / Open an issue in the repository
- Contactar directamente al autor / Contact the author directly

---

## 📄 Licencia / License

Este proyecto está bajo la licencia MIT. Ver archivo `LICENSE` para más detalles.

This project is under the MIT license. See the `LICENSE` file for more details.

---

## 🙏 Agradecimientos / Acknowledgments

- **Dispositivo AURA** por la adquisición de señales EEG
- **Comunidad de Lab Streaming Layer (LSL)** por el protocolo de streaming
- **Comunidad científica de neurotecnología** por las técnicas y metodologías

- **AURA Device** for EEG signal acquisition
- **Lab Streaming Layer (LSL) Community** for the streaming protocol
- **Neurotechnology scientific community** for techniques and methodologies

---

**Versión / Version:** 1.1  
**Última actualización / Last update:** Marzo 2026 / March 2026
**Autor / Author:** Rafael (PhD Candidate)
