# Utilidades y Scripts de Diagnóstico

Este directorio contiene scripts de utilidad para diagnóstico y validación de datos.

## Scripts Disponibles

### `step2_diagnose_acquisition.py`
**Propósito**: Diagnóstico de problemas de adquisición
- Analiza duraciones reales vs esperadas de fases
- Identifica gaps temporales
- Detecta problemas de secuencia de fases
- Útil para debugging de problemas de adquisición

**Uso**:
```bash
python analysis/utils/step2_diagnose_acquisition.py
```

### `validate_sampling_rate.py`
**Propósito**: Validación de frecuencia de muestreo
- Calcula frecuencia de muestreo real desde timestamps
- Identifica intervalos anómalos
- Detecta problemas de subsampling
- Útil para verificar calidad de datos

**Uso**:
```bash
python analysis/utils/validate_sampling_rate.py
```

### `verify_timers.py`
**Propósito**: Verificación de timers experimentales
- Compara duraciones reales vs esperadas
- Verifica transiciones entre fases
- Identifica problemas de sincronización
- Útil para validar protocolo experimental

**Uso**:
```bash
python analysis/utils/verify_timers.py
```

## Nota

Estos scripts son **opcionales** y se usan principalmente para:
- Debugging de problemas de adquisición
- Validación de calidad de datos
- Diagnóstico de inconsistencias

No son parte del pipeline principal de análisis, pero son útiles cuando se detectan problemas.
