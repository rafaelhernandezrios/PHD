# Proyecto de Procesamiento de Señales EEG

Este proyecto contiene scripts para procesar y visualizar señales EEG con múltiples etapas de filtrado.

## Requisitos / Requirements

- Python 3.7 o superior
- pip

## Instalación / Installation

### 1. Crear y activar el entorno virtual / Create and activate virtual environment

**Windows:**
```bash
# Crear el entorno virtual
python -m venv venv

# Activar el entorno virtual
venv\Scripts\activate
```

**Linux/Mac:**
```bash
# Crear el entorno virtual
python3 -m venv venv

# Activar el entorno virtual
source venv/bin/activate
```

### 2. Instalar dependencias / Install dependencies

```bash
pip install -r requirements.txt
```

## Uso / Usage

Ejecutar el script principal:
```bash
python plot_fp1.py
```

## Dependencias / Dependencies

- pandas: Manejo de datos CSV
- matplotlib: Visualización de gráficos
- numpy: Operaciones numéricas
- scipy: Filtros de señal
- PyWavelets: Transformadas wavelet

## Estructura del Proyecto / Project Structure

```
PHD/
├── data/
│   └── eeg.csv          # Datos EEG
├── plot_fp1.py          # Script principal de procesamiento
├── requirements.txt     # Dependencias del proyecto
├── README.md            # Este archivo
└── venv/                # Entorno virtual (no versionar)
```

## Pipeline de Procesamiento / Processing Pipeline

El script aplica las siguientes etapas de procesamiento:

1. **Señal Original**: Datos sin procesar del canal Fp1
2. **Filtro Bandpass**: 0.2-50 Hz
3. **Common Average Reference (CAR)**: Re-referenciación
4. **Filtro Stopband**: Eliminación de 60 Hz (ruido de red eléctrica)
5. **WAAF**: Wavelet-Assisted Adaptive Filter para remover artefactos oculares


