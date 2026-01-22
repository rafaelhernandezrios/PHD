# Baseline Recording - EEG Application

## Descripción / Description

Aplicación standalone para realizar grabación de baseline mientras se registran datos de EEG. La aplicación tiene dos fases:
1. **Ojos abiertos** (1.5 minutos): El usuario mira un punto de fijación en la pantalla
2. **Ojos cerrados** (1.5 minutos): El usuario mantiene los ojos cerrados

Standalone application to perform baseline recording while recording EEG data. The application has two phases:
1. **Eyes open** (1.5 minutes): User stares at a fixation point on screen
2. **Eyes closed** (1.5 minutes): User keeps eyes closed

## Características / Features

- **Dos fases automáticas**: Transición automática entre ojos abiertos y cerrados
- **Punto de fijación**: Círculo blanco con cruz para fijar la mirada
- **Duración**: 1.5 minutos por fase (90 segundos cada una)
- **Registro de EEG**: Guarda datos filtrados (~50 Hz) y raw (250 Hz)
- **Labels automáticos**: Etiqueta cada muestra con la fase correspondiente

## Requisitos / Requirements

- Python 3.9+
- Dependencias instaladas (ver `requirements.txt`)
- Dispositivo EEG AURA conectado vía LSL

## Uso / Usage

### 1. Ejecutar la aplicación / Run the application

```bash
python main_baseline.py
```

### 2. Configurar usuario / Set user name

1. Menú `File` → `Set User Name`
2. Ingresa el nombre del usuario
3. Se creará automáticamente la carpeta `data_[nombre_usuario]`

### 3. Conectar al EEG / Connect to EEG

1. Menú `File` → `Connect to EEG`
2. Espera a que aparezca el mensaje "Connected" en la barra de estado

### 4. Iniciar la grabación / Start recording

1. Haz clic en el botón "Start Baseline Recording"
2. **Fase 1 - Ojos abiertos**:
   - Mantén los ojos abiertos
   - Mira el punto blanco en el centro de la pantalla
   - Intenta mantenerte relajado y evitar parpadear excesivamente
   - Duración: 1.5 minutos

3. **Transición automática**:
   - Después de 1.5 minutos, aparecerá un mensaje de transición
   - Tendrás 2 segundos para prepararte

4. **Fase 2 - Ojos cerrados**:
   - Cierra los ojos
   - Mantén los ojos cerrados hasta que termine la grabación
   - Duración: 1.5 minutos

5. **Finalización**:
   - Los datos se guardan automáticamente al finalizar
   - Puedes abrir los ojos cuando aparezca el mensaje de finalización

### 5. Guardar datos / Save data

Los datos se guardan automáticamente al finalizar la grabación. También puedes guardar manualmente:
- Menú `File` → `Save Data`

## Archivos generados / Generated files

Se generan 2 archivos CSV en la carpeta `data_[usuario]/`:

1. **`baseline_filtered_[timestamp].csv`**: Datos filtrados (~50 Hz)
   - Columnas: `timestamp`, `phase_label`, `channel_0` a `channel_7`
   - Labels: `baseline_eyes_open`, `baseline_eyes_closed`

2. **`baseline_raw_[timestamp].csv`**: Datos raw (250 Hz)
   - Misma estructura que filtered pero a frecuencia completa

## Estructura de datos / Data structure

### Labels de fase / Phase labels

- `baseline_eyes_open`: Fase de ojos abiertos (1.5 minutos)
- `baseline_eyes_closed`: Fase de ojos cerrados (1.5 minutos)

### Parámetros de la grabación / Recording parameters

- **Duración total**: 3 minutos (180 segundos)
- **Fase 1 (Ojos abiertos)**: 90 segundos (1.5 minutos)
- **Fase 2 (Ojos cerrados)**: 90 segundos (1.5 minutos)
- **Transición entre fases**: 2 segundos

## Notas / Notes

- La aplicación registra datos automáticamente cuando la grabación está en ejecución
- Los datos se guardan al finalizar la grabación o manualmente desde el menú
- Si cierras la aplicación sin guardar, se te preguntará si deseas guardar los datos
- El punto de fijación desaparece durante la fase de ojos cerrados
- La transición entre fases es automática, no requiere intervención del usuario
