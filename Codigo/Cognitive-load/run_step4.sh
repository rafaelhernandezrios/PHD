#!/usr/bin/env bash
set -euo pipefail

# Run step4 cognitive-load analysis with a non-GUI matplotlib backend
# to avoid macOS AppKit crashes in headless/non-interactive contexts.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PYTHON="$SCRIPT_DIR/venv/bin/python"
STEP4_SCRIPT="$SCRIPT_DIR/analysis/pipeline/step4_cognitive_load_cleaned.py"

if [[ ! -x "$VENV_PYTHON" ]]; then
  echo "Error: Python del venv no encontrado en: $VENV_PYTHON"
  echo "Error: venv Python not found at: $VENV_PYTHON"
  exit 1
fi

if [[ ! -f "$STEP4_SCRIPT" ]]; then
  echo "Error: Script no encontrado en: $STEP4_SCRIPT"
  echo "Error: Script not found at: $STEP4_SCRIPT"
  exit 1
fi

export MPLBACKEND=Agg
export MPLCONFIGDIR="$SCRIPT_DIR/.mplconfig"
mkdir -p "$MPLCONFIGDIR"

echo "Ejecutando step4 con backend Agg..."
echo "Running step4 with Agg backend..."
"$VENV_PYTHON" "$STEP4_SCRIPT"

echo "Listo. Resultados en: $SCRIPT_DIR/output/analysis_output"
echo "Done. Results in: $SCRIPT_DIR/output/analysis_output"
