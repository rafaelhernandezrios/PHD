# Documentación: Uso de Datos Actuales para Análisis

## Estado Actual de los Datos

### Datos Disponibles

1. **Data-Experimento-Rafa** (7 sujetos)
   - Promedio: ~1,600 muestras por sujeto
   - Duración: ~6-7 segundos
   - Problema: Datos muy cortos (~5% de lo esperado)

2. **data_test1** (1 sujeto)
   - Total: 260,744 muestras
   - Duración: 13.29 minutos
   - Mejora: 163x más datos que el promedio anterior

### Inconsistencias Identificadas

#### 1. Frecuencia de Muestreo
- **Esperado**: 50 Hz (con subsampling de 1 cada 5)
- **Detectado**: ~415 Hz
- **Causa**: El subsampling no se está aplicando correctamente
- **Impacto**: Archivos más grandes, pero datos más completos

#### 2. Duraciones de Fases
- **Baseline**: ~720 segundos (esperado: 90 seg) → 8x más largo
- **Low Load**: ~1442 segundos (esperado: 180 seg) → 8x más largo
- **High Load**: ~953 segundos (esperado: 180 seg) → 5.3x más largo
- **Causa**: Los timers no están deteniendo las fases correctamente, o el logging continúa después de que la fase termina

#### 3. Intervalos Temporales
- Intervalo mediano: 0.00241 segundos
- Hay gaps grandes (hasta 103 segundos)
- **Causa**: Posibles problemas de logging o timestamps

## Recomendaciones para Análisis

### ✅ Usar los Datos Actuales

**A pesar de las inconsistencias, los datos son utilizables para análisis porque:**

1. **Calidad de Señal**: No hay valores NaN, rangos razonables
2. **Cobertura Completa**: Todas las fases están presentes
3. **Suficiente para Análisis**: 489 ventanas válidas para cognitive load
4. **Hipótesis Cumplida**: High Load > Low Load (p < 0.05)

### ⚠️ Consideraciones Importantes

1. **Frecuencia de Muestreo**: 
   - Usar la frecuencia detectada (~415 Hz) en lugar de la esperada (250 Hz)
   - O aplicar subsampling post-hoc si se necesita 50 Hz

2. **Duraciones de Fases**:
   - Las fases son más largas de lo esperado
   - Esto puede ser beneficioso (más datos) o problemático (fatiga del sujeto)
   - Considerar usar solo los primeros X segundos de cada fase

3. **Análisis Temporal**:
   - Usar timestamps reales en lugar de asumir frecuencia constante
   - Considerar gaps en el análisis

### 📊 Estrategia de Análisis

1. **Preprocesamiento**:
   - Aplicar filtros (bandpass, notch, CAR) como se hace actualmente
   - Usar frecuencia detectada para cálculos espectrales

2. **Ventanas de Análisis**:
   - Usar ventanas deslizantes de 2 segundos
   - 50% overlap para suavizar resultados
   - Filtrar ventanas con gaps grandes

3. **Comparaciones**:
   - Comparar High vs Low Load (hipótesis principal)
   - Comparar Baseline vs Cognitive Load
   - Considerar efectos temporales (inicio vs final de fase)

## Correcciones Implementadas

### 1. Subsampling Corregido
- Contador inicializado correctamente
- Factor de subsampling configurable (`_subsampling_factor = 5`)
- Logs de diagnóstico agregados

### 2. Validación de Frecuencia
- Script `validate_sampling_rate.py` creado
- Detecta frecuencia real automáticamente
- Identifica gaps y problemas temporales

### 3. Verificación de Timers
- Script `verify_timers.py` creado
- Verifica duraciones de fases
- Detecta problemas de transición

## Próximos Pasos

1. **Probar correcciones**: Ejecutar nuevo experimento y verificar que el subsampling funcione
2. **Validar timers**: Verificar que las fases se detengan correctamente
3. **Análisis con datos actuales**: Continuar usando data_test1 para análisis
4. **Comparar resultados**: Una vez corregido, comparar con datos anteriores
