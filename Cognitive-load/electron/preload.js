/**
 * preload.js — contextBridge security layer
 * Exposes only the needed IPC methods to the renderer.
 * Never expose ipcRenderer directly.
 */

const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('electronAPI', {
  // Renderer → Main (invoke = async request/response)
  connectDevice:    ()       => ipcRenderer.invoke('connect-device'),
  disconnectDevice: ()       => ipcRenderer.invoke('disconnect-device'),
  startSetup:       (userId) => ipcRenderer.invoke('start-setup', userId),
  startBaseline:    ()       => ipcRenderer.invoke('start-baseline'),
  startLowLoad:     ()       => ipcRenderer.invoke('start-low-load'),
  startHighLoad:    ()       => ipcRenderer.invoke('start-high-load'),
  setPhaseDurations: (durations) => ipcRenderer.invoke('set-phase-durations', durations),
  saveData:         ()       => ipcRenderer.invoke('save-data'),
  scanLSL:          ()       => ipcRenderer.invoke('scan-lsl'),
  logRendererError: (payload) => ipcRenderer.invoke('renderer-log-error', payload),

  // Main → Renderer (push events from Python)
  onConnectionStatus: (cb) => ipcRenderer.on('connection-status', (_e, ...a) => cb(...a)),
  onPhaseChanged:     (cb) => ipcRenderer.on('phase-changed',     (_e, ...a) => cb(...a)),
  onTimerUpdate:      (cb) => ipcRenderer.on('timer-update',      (_e, ...a) => cb(...a)),
  onRatioUpdate:      (cb) => ipcRenderer.on('ratio-update',      (_e, ...a) => cb(...a)),
  onSampleCount:      (cb) => ipcRenderer.on('sample-count',      (_e, ...a) => cb(...a)),
  onDeviceInfo:       (cb) => ipcRenderer.on('device-info',       (_e, ...a) => cb(...a)),
  onPlotSample:       (cb) => ipcRenderer.on('plot-sample',       (_e, ...a) => cb(...a)),
  onPhaseDurations:   (cb) => ipcRenderer.on('phase-durations',   (_e, ...a) => cb(...a)),
});
