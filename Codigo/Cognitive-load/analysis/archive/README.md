# Archivo de Scripts Antiguos

Este directorio contiene scripts antiguos o redundantes que ya no se usan en el pipeline principal.

## Scripts Archivados

### Scripts Reemplazados por el Pipeline
- `analyze_all_subjects.py` - Reemplazado por `pipeline/step4_cognitive_load_cleaned.py`
- `analyze_cognitive_load.py` - Versión antigua, reemplazada por el pipeline
- `analyze_cognitive_load_test1.py` - Específico para test1, usar pipeline general
- `analyze_test1.py` - Específico para test1, usar `pipeline/step1_explore_data.py`
- `analyze_latest_test1.py` - Específico para test1

### Scripts con Métodos Descontinuados
- `analyze_with_waaf.py` - WAAF ya no se usa (no mejoró resultados)

### Scripts Opcionales/Especializados
- `detailed_comparison.py` - Comparación detallada entre sujetos (opcional)
- `plot_channels_by_label.py` - Visualización de canales (ya incluido en step2)

## Nota

Estos scripts se mantienen por referencia histórica o para casos especiales. 
**No se recomienda usarlos** para análisis nuevos. Usar el pipeline en `pipeline/` en su lugar.

Si necesitas alguna funcionalidad específica de estos scripts, considera:
1. Verificar si ya está implementada en el pipeline
2. Integrarla en el pipeline si es útil
3. Crear un script nuevo en `utils/` si es solo para diagnóstico
