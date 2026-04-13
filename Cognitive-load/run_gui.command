#!/bin/zsh
set -euo pipefail

# One-click launcher for Cognitive-load Electron GUI on macOS.
# It ensures venv + Python deps + npm deps, then starts Electron
# using the venv Python interpreter (required for pylsl).

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "==> Cognitive-load launcher"
echo "Project: $SCRIPT_DIR"

if [[ ! -d "venv" ]]; then
  echo "==> Creating venv with python3..."
  python3 -m venv venv
fi

echo "==> Activating venv..."
source "venv/bin/activate"

echo "==> Installing/updating Python dependencies..."
python3 -m pip install --upgrade pip >/dev/null
python3 -m pip install -r "requirements.txt"

cd "electron"

if [[ ! -d "node_modules" ]]; then
  echo "==> Installing Electron dependencies..."
  npm install
fi

echo "==> Starting Electron with venv python..."
# Some environments (IDE/CI) export this and break Electron GUI startup.
unset ELECTRON_RUN_AS_NODE
mkdir -p "electron/logs"
echo "==> Logs:"
echo "    bridge:  $SCRIPT_DIR/electron/logs/eeg_bridge.log"
echo "    electron:$SCRIPT_DIR/electron/logs/electron_main.log"
PYTHON_PATH="../venv/bin/python" npm start
