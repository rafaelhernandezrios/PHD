# Análisis de Cognitive Load

Este directorio contiene todos los scripts para el análisis de datos EEG y cálculo de cognitive load.

## Estructura

```
analysis/
├── pipeline/          # Scripts principales del pipeline (USAR ESTOS)
├── utils/             # Scripts de utilidad/diagnóstico (opcionales)
├── archive/           # Scripts antiguos/redundantes (no usar)
└── DOCUMENTACION_DATOS.md  # Documentación de datos
```

## Pipeline Principal

**Usar estos scripts en orden:**

1. `pipeline/step1_explore_data.py` - Exploración inicial
2. `pipeline/step2_analyze_individual_subjects.py` - Análisis individual
3. `pipeline/step3_detect_artifacts.py` - Detección de artefactos
4. `pipeline/step4_cognitive_load_cleaned.py` - Cálculo final ⭐

Ver `pipeline/README.md` para más detalles.

## Utilidades

Scripts opcionales para diagnóstico:
- `utils/step2_diagnose_acquisition.py` - Diagnóstico de adquisición
- `utils/validate_sampling_rate.py` - Validación de frecuencia
- `utils/verify_timers.py` - Verificación de timers

Ver `utils/README.md` para más detalles.

## Archivo

Scripts antiguos mantenidos por referencia. **No usar para análisis nuevos.**

Ver `archive/README.md` para más detalles.

## Proceso Completo

El pipeline completo incluye:

1. **Adquisición** (apps/) - Captura de datos EEG
2. **Preprocesamiento**:
   - Filtro Notch (60 Hz)
   - Filtro Bandpass (1-40 Hz)
   - Common Average Reference (CAR)
   - Normalización Z-score
3. **Limpieza de Artefactos**:
   - Detección (Z-score, IQR, Amplitud)
   - Supresión (Interpolación)
4. **Cálculo de Cognitive Load**:
   - Theta (4-7 Hz) en Fz
   - Alpha (8-12 Hz) en Pz
   - Ratio = Theta_Fz / Alpha_Pz
5. **Análisis Estadístico**:
   - Comparación High vs Low Load
   - Verificación de hipótesis

## Resultados

Todos los resultados se guardan en `output/analysis_output/`:
- Gráficos individuales
- Reportes CSV
- Visualizaciones comparativas
- Resumen de hipótesis
