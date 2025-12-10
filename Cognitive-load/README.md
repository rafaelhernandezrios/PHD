# Plataforma de Experimentación EEG - Carga Cognitiva

Plataforma completa para experimentación EEG en tiempo real con análisis de carga cognitiva utilizando el dispositivo AURA.

## 📋 Descripción

Sistema de adquisición y análisis de señales EEG diseñado para experimentos de carga cognitiva. La plataforma permite:

- **Adquisición en tiempo real** de señales EEG vía LSL (Lab Streaming Layer)
- **Procesamiento de señal** con filtrado digital y análisis espectral
- **Protocolo experimental estructurado** con múltiples fases
- **Visualización en tiempo real** de señales y métricas de carga cognitiva
- **Tareas cognitivas integradas** (Stroop, lectura pasiva)
- **Data logging** organizado por usuario

## 🎯 Características Principales

### Adquisición y Procesamiento
- Conexión vía LSL al dispositivo AURA (8 canales, 250 Hz)
- Filtrado digital: Notch 60 Hz + Pasabanda 1-40 Hz
- Análisis espectral en tiempo real (Método de Welch)
- Cálculo de bandpower: Theta (4-7 Hz) y Alpha (8-12 Hz)
- Índice de carga cognitiva: Ratio Theta_Fz / Alpha_Pz

### Protocolo Experimental
1. **Setup**: Verificación de calidad de señal
2. **Baseline**: 90s ojos abiertos + 90s ojos cerrados
3. **Baja Carga**: Lectura pasiva de texto (3 min)
4. **Alta Carga**: Tarea Stroop (3 min)
5. **Análisis**: Procesamiento y visualización de resultados

### Interfaz Gráfica
- Dashboard científico con tema oscuro
- Visualización de 8 canales EEG individuales
- Gráfico en tiempo real del ratio de carga cognitiva
- Tareas cognitivas integradas con feedback inmediato

## 🚀 Instalación

### Requisitos
- Python 3.9 o superior
- Dispositivo AURA con drivers instalados
- LSL (Lab Streaming Layer) configurado

### Pasos de Instalación

1. **Clonar el repositorio**
```bash
git clone https://github.com/tu-usuario/Cognitive-load.git
cd Cognitive-load
```

2. **Crear entorno virtual**
```bash
python -m venv venv
```

3. **Activar entorno virtual**

**Windows (PowerShell):**
```powershell
.\venv\Scripts\Activate.ps1
```

**Windows (CMD):**
```cmd
venv\Scripts\activate.bat
```

**Linux/Mac:**
```bash
source venv/bin/activate
```

4. **Instalar dependencias**
```bash
pip install -r requirements.txt
```

## 📖 Uso

### Ejecutar la Aplicación

```bash
python main.py
```

### Flujo de Uso

1. **Conectar Dispositivo**
   - Asegúrate de que el dispositivo AURA esté encendido y transmitiendo vía LSL
   - Haz clic en "Conectar AURA"
   - Verifica que aparezca "Conectado" en verde

2. **Iniciar Setup**
   - Haz clic en "Iniciar Setup"
   - Ingresa el nombre o ID del usuario cuando se solicite
   - Verifica la calidad de la señal en los gráficos de los 8 canales

3. **Ejecutar Protocolo**
   - **Baseline**: Haz clic en "Iniciar Baseline"
     - Mantén los ojos abiertos durante 90 segundos
     - Luego cierra los ojos durante 90 segundos
   - **Baja Carga**: Haz clic en "Iniciar Baja Carga"
     - Lee el texto que aparece en pantalla de manera pasiva
   - **Alta Carga**: Haz clic en "Iniciar Alta Carga"
     - Realiza la tarea Stroop:
       - Presiona **R** para Rojo
       - Presiona **A** para Azul
       - Presiona **V** para Verde
       - Presiona **Y** para Amarillo
       - Identifica el **COLOR de la tinta**, no la palabra escrita

4. **Guardar Datos**
   - Al finalizar, haz clic en "Guardar Datos"
   - Los datos se guardarán en `data_[nombre_usuario]/eeg_data_[timestamp].csv`

## 📁 Estructura del Proyecto

```
Cognitive-load/
├── main.py                      # Punto de entrada de la aplicación
├── signal_worker.py             # Worker thread para adquisición LSL
├── experiment_logic.py           # Máquina de estados del protocolo
├── ui_main.py                   # Interfaz gráfica PyQt5
├── requirements.txt             # Dependencias del proyecto
├── FLUJO_CARGA_COGNITIVA.md    # Documentación técnica detallada
├── README.md                    # Este archivo
├── .gitignore                   # Archivos ignorados por Git
└── data_*/                      # Carpetas de datos por usuario (no versionadas)
    └── eeg_data_*.csv           # Archivos CSV con datos experimentales
```

## 📊 Formato de Datos

Los archivos CSV guardados contienen las siguientes columnas:

- `timestamp`: Timestamp de la muestra
- `phase`: Fase del experimento (setup, baseline_eyes_open, etc.)
- `label`: Etiqueta descriptiva de la fase (setup, baseline_eyes_open, low_cognitive_load, high_cognitive_load, etc.)
- `channel_0` a `channel_7`: Valores de los 8 canales EEG (filtrados, en unidades del dispositivo)

## 🔧 Configuración Técnica

### Parámetros del Sistema

| Parámetro | Valor |
|-----------|-------|
| Tasa de muestreo | 250 Hz |
| Canales EEG | 8 |
| Filtro Notch | 60 Hz, Q=30 |
| Filtro Pasabanda | 1-40 Hz, orden 4 |
| Ventana de análisis | 2 segundos (500 muestras) |
| Banda Theta | 4-7 Hz (Canal Fz) |
| Banda Alpha | 8-12 Hz (Canal Pz) |
| Frecuencia de cálculo | 1 Hz |

### Mapeo de Canales

- **Canal 0**: Fz (Frontal) - Usado para análisis Theta
- **Canal 1-3**: Canales adicionales
- **Canal 4**: Pz (Parietal) - Usado para análisis Alpha
- **Canal 5-7**: Canales adicionales

## 📚 Documentación

Para información técnica detallada sobre el flujo de procesamiento, consulta:
- [FLUJO_CARGA_COGNITIVA.md](FLUJO_CARGA_COGNITIVA.md) - Documentación completa del sistema

## 🛠️ Tecnologías Utilizadas

- **Python 3.9+**
- **PyQt5** - Interfaz gráfica
- **pyqtgraph** - Visualización en tiempo real
- **NumPy** - Procesamiento numérico
- **SciPy** - Filtros digitales y análisis espectral
- **Pandas** - Manejo de datos y exportación CSV
- **pylsl** - Comunicación con Lab Streaming Layer

## 📝 Licencia

Este proyecto está bajo la licencia MIT. Ver archivo `LICENSE` para más detalles.

## 👥 Contribuciones

Las contribuciones son bienvenidas. Por favor:

1. Fork el proyecto
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## 📧 Contacto

Para preguntas o soporte, por favor abre un issue en el repositorio.

## 🙏 Agradecimientos

- Dispositivo AURA por la adquisición de señales EEG
- Comunidad de Lab Streaming Layer (LSL)
- Comunidad científica de neurotecnología

---

**Versión:** 1.0  
**Última actualización:** Diciembre 2024

