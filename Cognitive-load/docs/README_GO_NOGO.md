# Go/No-Go Task - EEG Recording Application

## Descripción / Description

Aplicación standalone para realizar la tarea Go/No-Go mientras se registran datos de EEG. La tarea dura exactamente 2 minutos y guarda automáticamente los datos en archivos CSV.

Standalone application to perform the Go/No-Go task while recording EEG data. The task lasts exactly 2 minutes and automatically saves data to CSV files.

## Características / Features

- **Interfaz gráfica Go/No-Go**: Círculos verdes (Go) y cuadrados rojos (No-Go)
- **Duración**: 2 minutos exactos
- **Registro de EEG**: Guarda datos filtrados (~50 Hz) y raw (250 Hz)
- **Eventos de respuesta**: Guarda tiempos de reacción y precisión
- **Estadísticas en tiempo real**: Muestra aciertos y errores

## Requisitos / Requirements

- Python 3.9+
- Dependencias instaladas (ver `requirements.txt`)
- Dispositivo EEG AURA conectado vía LSL

## Uso / Usage

### 1. Ejecutar la aplicación / Run the application

```bash
python main_go_nogo.py
```

### 2. Configurar usuario / Set user name

1. Menú `File` → `Set User Name`
2. Ingresa el nombre del usuario
3. Se creará automáticamente la carpeta `data_[nombre_usuario]`

### 3. Conectar al EEG / Connect to EEG

1. Menú `File` → `Connect to EEG`
2. Espera a que aparezca el mensaje "Connected" en la barra de estado

### 4. Iniciar la tarea / Start task

1. Haz clic en el botón "Start Task"
2. La tarea comenzará automáticamente después de 1 segundo
3. **Instrucciones**:
   - Presiona **ESPACIO** cuando veas un **círculo verde** (Go)
   - **NO presiones** cuando veas un **cuadrado rojo** (No-Go)

### 5. Guardar datos / Save data

Los datos se guardan automáticamente al finalizar la tarea. También puedes guardar manualmente:
- Menú `File` → `Save Data`

## Archivos generados / Generated files

Se generan 3 archivos CSV en la carpeta `data_[usuario]/`:

1. **`go_nogo_filtered_[timestamp].csv`**: Datos filtrados (~50 Hz)
   - Columnas: `timestamp`, `trial_label`, `channel_0` a `channel_7`
   - Labels: `go_correct`, `go_incorrect`, `no_go_correct`, `no_go_incorrect`

2. **`go_nogo_raw_[timestamp].csv`**: Datos raw (250 Hz)
   - Misma estructura que filtered pero a frecuencia completa

3. **`go_nogo_events_[timestamp].csv`**: Eventos de respuesta
   - Columnas: `timestamp`, `stimulus_label`, `is_correct`, `reaction_time`

## Estructura de datos / Data structure

### Labels de trial / Trial labels

- `go_correct`: Respuesta correcta a estímulo Go
- `go_incorrect`: Respuesta incorrecta a estímulo Go (no presionó)
- `no_go_correct`: Respuesta correcta a estímulo No-Go (no presionó)
- `no_go_incorrect`: Respuesta incorrecta a estímulo No-Go (presionó - false alarm)

### Parámetros de la tarea / Task parameters

- **Duración**: 120 segundos (2 minutos)
- **Intervalo entre estímulos**: 1-2 segundos (variable)
- **Ratio Go/No-Go**: 70% Go, 30% No-Go
- **Tiempo máximo de respuesta**: 1.5 segundos

## Notas / Notes

- La aplicación registra datos automáticamente cuando la tarea está en ejecución
- Los datos se guardan al finalizar la tarea o manualmente desde el menú
- Si cierras la aplicación sin guardar, se te preguntará si deseas guardar los datos
