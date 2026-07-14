# Pipeline de Análisis de Cognitive Load

Este directorio contiene los scripts principales del pipeline de análisis de cognitive load.

## Scripts del Pipeline

Ejecutar en orden secuencial:

### 1. `step1_explore_data.py`
**Propósito**: Exploración inicial de datos
- Lista todos los archivos CSV disponibles
- Muestra estadísticas básicas por sujeto
- Cuenta muestras por fase
- Identifica datos faltantes o problemáticos

**Uso**:
```bash
python analysis/pipeline/step1_explore_data.py
```

### 2. `step2_analyze_individual_subjects.py`
**Propósito**: Análisis individual por sujeto
- Revisa cada sujeto por separado
- Visualiza señales por canal (Fz y Pz)
- Verifica preprocesamiento (filtros, CAR)
- Calcula métricas básicas

**Uso**:
```bash
python analysis/pipeline/step2_analyze_individual_subjects.py
```

### 3. `step3_detect_artifacts.py`
**Propósito**: Detección y supresión de artefactos
- Detecta picos/outliers en las señales
- Visualiza artefactos detectados
- Aplica métodos de supresión (interpolación)
- Compara resultados antes/después

**Uso**:
```bash
python analysis/pipeline/step3_detect_artifacts.py
```

### 4. `step4_cognitive_load_cleaned.py`
**Propósito**: Cálculo final de cognitive load con datos limpiados
- Aplica limpieza de artefactos a todos los datos
- Recalcula ratios de cognitive load
- Compara resultados antes/después de la limpieza
- Genera visualizaciones y reportes finales
- Verifica hipótesis: High Load > Low Load

**Uso**:
```bash
python analysis/pipeline/step4_cognitive_load_cleaned.py
```

## Pipeline Completo

Para ejecutar todo el pipeline en orden:

```bash
python analysis/pipeline/step1_explore_data.py
python analysis/pipeline/step2_analyze_individual_subjects.py
python analysis/pipeline/step3_detect_artifacts.py
python analysis/pipeline/step4_cognitive_load_cleaned.py
```

## Resultados

Los resultados se guardan en `output/analysis_output/`:
- Gráficos individuales por sujeto
- Reportes CSV con métricas
- Visualizaciones comparativas
- Resumen final de hipótesis
