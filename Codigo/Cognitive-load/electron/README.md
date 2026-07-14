# Electron UI for Cognitive-load

## Requisitos

- Python 3.9+
- Dependencias Python del proyecto instaladas (`pip install -r ../requirements.txt`)
- Node.js 18+

## Ejecutar

1. Entra a esta carpeta:
   - `cd Cognitive-load/electron`
2. Instala dependencias:
   - `npm install`
3. Inicia la app:
   - `npm start`

## Seleccionar Python (opcional)

Si quieres usar el Python del `venv` del proyecto:

- macOS/Linux:
  - `PYTHON_PATH=../venv/bin/python npm start`

Esto hace que Electron lance `eeg_bridge.py` con ese intérprete.
