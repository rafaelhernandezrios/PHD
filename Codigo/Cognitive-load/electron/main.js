/**
 * main.js — Electron entry point
 * Launches the renderer and bridges IPC calls to the Python backend.
 *
 * Structure:
 *   main.js          ← this file
 *   preload.js       ← contextBridge (security layer)
 *   index.html       ← renderer UI
 *   eeg_bridge.py    ← JSON bridge to existing Python backend
 *
 * Install deps:
 *   npm install electron python-shell
 *
 * Run:
 *   npx electron .
 */

let electronMain;
try {
  // Preferred entry in modern Electron versions
  electronMain = require('electron/main');
} catch (_err) {
  // Backward-compatible fallback
  electronMain = require('electron');
}
const { app, BrowserWindow, ipcMain, dialog } = electronMain;
const path = require('path');
const fs = require('fs');
const { PythonShell } = require('python-shell');

if (!app || !BrowserWindow || !ipcMain) {
  console.error('[fatal] Electron main process API is unavailable.');
  console.error('[fatal] Ensure app is launched with `electron .` and not `node main.js`.');
  process.exit(1);
}

let mainWindow;
let pyShell = null;  // persistent Python process
let lastScanAt = 0;
let isConnecting = false;
const logDir = path.join(__dirname, 'logs');
fs.mkdirSync(logDir, { recursive: true });
const mainLogPath = path.join(logDir, 'electron_main.log');

function mainLog(message, data = null) {
  const payload = {
    ts: new Date().toISOString(),
    message,
    data,
  };
  fs.appendFileSync(mainLogPath, JSON.stringify(payload) + '\n', 'utf8');
}

// ─────────────────────────────────────────────────────────────────────
// WINDOW
// ─────────────────────────────────────────────────────────────────────
function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1280,
    height: 820,
    minWidth: 960,
    minHeight: 600,
    titleBarStyle: 'hidden',          // custom titlebar in HTML
    trafficLightPosition: { x: -999, y: -999 }, // hide native dots (drawn in HTML)
    backgroundColor: '#080c10',
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false,
    },
  });

  mainWindow.loadFile('index.html');

  // Open DevTools in dev mode
  if (process.env.NODE_ENV === 'development') {
    mainWindow.webContents.openDevTools({ mode: 'detach' });
  }
}

app.whenReady().then(createWindow);
app.on('window-all-closed', () => { if (process.platform !== 'darwin') app.quit(); });
app.on('activate', () => { if (BrowserWindow.getAllWindows().length === 0) createWindow(); });
app.on('before-quit', () => {
  sendToPython('shutdown');
  stopPython();
});

// ─────────────────────────────────────────────────────────────────────
// PYTHON BRIDGE
// Spawns eeg_bridge.py which wraps your existing modules.
// Messages are NDJSON lines (one JSON object per line).
// ─────────────────────────────────────────────────────────────────────
function startPython() {
  if (pyShell) return;

  const projectVenvPython = path.resolve(__dirname, '..', 'venv', 'bin', 'python');
  const pythonPath =
    process.env.PYTHON_PATH ||
    (fs.existsSync(projectVenvPython) ? projectVenvPython : 'python3');
  mainLog('start_python', { pythonPath });

  pyShell = new PythonShell(path.join(__dirname, 'eeg_bridge.py'), {
    mode: 'text',
    pythonPath,
    pythonOptions: ['-u'],
  });

  // Python → Renderer
  pyShell.on('message', (line) => {
    let msg;
    try {
      msg = JSON.parse(line);
    } catch (_e) {
      // Ignore non-JSON stdout lines from backend internals.
      mainLog('python_stdout_non_json', { line });
      return;
    }
    mainLog('python_message', msg);
    if (!mainWindow) return;
    const { event, data } = msg;
    switch (event) {
      case 'connection_status':
        mainWindow.webContents.send('connection-status', data.connected, data.message);
        break;
      case 'phase_changed':
        mainWindow.webContents.send('phase-changed', data.phase, data.message);
        break;
      case 'timer_update':
        mainWindow.webContents.send('timer-update', data.time);
        break;
      case 'ratio_update':
        mainWindow.webContents.send('ratio-update', data.ratio, data.theta, data.alpha);
        break;
      case 'sample_count':
        mainWindow.webContents.send('sample-count', data.count);
        break;
      case 'device_info':
        mainWindow.webContents.send('device-info', data);
        break;
      case 'phase_durations':
        mainWindow.webContents.send('phase-durations', data);
        break;
      case 'plot_sample':
        mainWindow.webContents.send('plot-sample', data.raw, data.filtered, data.timestamp);
        break;
      case 'save_done':
        dialog.showMessageBox(mainWindow, {
          type: 'info',
          title: 'Data Saved',
          message: `Saved ${data.rows} rows to:\n${data.filepath}`,
        });
        break;
      case 'save_error':
        dialog.showErrorBox('Save Error', data.error);
        break;
      case 'ecological_state':
        mainWindow.webContents.send('ecological-state', data);
        break;
      case 'ecological_error':
        mainWindow.webContents.send('ecological-error', data);
        break;
      default:
        console.warn('[python]', msg);
    }
  });

  pyShell.on('stderr', (line) => console.error('[python stderr]', line));
  pyShell.on('stderr', (line) => {
    console.error('[python stderr]', line);
    mainLog('python_stderr', { line });
  });
  pyShell.on('error', (err) => {
    console.error('[python error]', err);
    mainLog('python_error', { error: String(err) });
  });
}

function stopPython() {
  if (pyShell) { pyShell.kill(); pyShell = null; }
}

function sendToPython(cmd, payload = {}) {
  mainLog('send_to_python', { cmd, payload });
  if (!pyShell) {
    console.warn('[main] Python not running — command ignored:', cmd);
    return;
  }
  pyShell.send(JSON.stringify({ cmd, ...payload }));
}

// ─────────────────────────────────────────────────────────────────────
// IPC HANDLERS  (renderer → main → python)
// ─────────────────────────────────────────────────────────────────────
ipcMain.handle('connect-device', async () => {
  if (isConnecting) return;
  isConnecting = true;
  startPython();
  sendToPython('connect');
  setTimeout(() => { isConnecting = false; }, 1500);
});

ipcMain.handle('disconnect-device', async () => {
  sendToPython('shutdown');
  stopPython();
});

ipcMain.handle('start-setup', async (_event, userId) => {
  sendToPython('start_setup', { user: userId });
});

ipcMain.handle('start-baseline', async () => {
  sendToPython('start_baseline');
});

ipcMain.handle('start-low-load', async () => {
  sendToPython('start_low_load');
});

ipcMain.handle('start-high-load', async () => {
  sendToPython('start_high_load');
});

ipcMain.handle('set-phase-durations', async (_event, durations) => {
  startPython();
  sendToPython('set_phase_durations', durations || {});
});

ipcMain.handle('save-data', async () => {
  sendToPython('save_data');
});

ipcMain.handle('ecological-start', async (_event, modality) => {
  startPython();
  sendToPython('ecological_start', { modality: modality || '' });
});

ipcMain.handle('ecological-stop', async () => {
  sendToPython('ecological_stop');
});

ipcMain.handle('scan-lsl', async () => {
  const now = Date.now();
  if (now - lastScanAt < 800) return;
  lastScanAt = now;
  startPython();
  sendToPython('scan_lsl');
});

ipcMain.handle('renderer-log-error', async (_event, payload) => {
  mainLog('renderer_error', payload || {});
});
