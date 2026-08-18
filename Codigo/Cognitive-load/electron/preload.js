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
  ecologicalStart:  (modality) => ipcRenderer.invoke('ecological-start', modality),
  ecologicalStop:   ()       => ipcRenderer.invoke('ecological-stop'),
  scanLSL:          ()       => ipcRenderer.invoke('scan-lsl'),
  logRendererError: (payload) => ipcRenderer.invoke('renderer-log-error', payload),

  // ERP support: clock alignment + behavioural/event logging
  clockPing:        (payload) => ipcRenderer.invoke('clock-ping', payload),
  clockOffset:      (payload) => ipcRenderer.invoke('clock-offset', payload),
  stroopTrial:      (trial)   => ipcRenderer.invoke('stroop-trial', trial),
  mark:             (payload) => ipcRenderer.invoke('mark', payload),

  // Main → Renderer (push events from Python)
  onConnectionStatus: (cb) => ipcRenderer.on('connection-status', (_e, ...a) => cb(...a)),
  onPhaseChanged:     (cb) => ipcRenderer.on('phase-changed',     (_e, ...a) => cb(...a)),
  onTimerUpdate:      (cb) => ipcRenderer.on('timer-update',      (_e, ...a) => cb(...a)),
  onRatioUpdate:      (cb) => ipcRenderer.on('ratio-update',      (_e, ...a) => cb(...a)),
  onSampleCount:      (cb) => ipcRenderer.on('sample-count',      (_e, ...a) => cb(...a)),
  onDeviceInfo:       (cb) => ipcRenderer.on('device-info',       (_e, ...a) => cb(...a)),
  onPlotSample:       (cb) => ipcRenderer.on('plot-sample',       (_e, ...a) => cb(...a)),
  onPhaseDurations:   (cb) => ipcRenderer.on('phase-durations',   (_e, ...a) => cb(...a)),
  onEcologicalState:  (cb) => ipcRenderer.on('ecological-state',  (_e, ...a) => cb(...a)),
  onEcologicalError:  (cb) => ipcRenderer.on('ecological-error',  (_e, ...a) => cb(...a)),
  onClockPong:        (cb) => ipcRenderer.on('clock-pong',        (_e, ...a) => cb(...a)),
  onClockSyncOk:      (cb) => ipcRenderer.on('clock-sync-ok',     (_e, ...a) => cb(...a)),
  onAcqHealth:        (cb) => ipcRenderer.on('acq-health',        (_e, ...a) => cb(...a)),
  onStreamStatus:     (cb) => ipcRenderer.on('stream-status',     (_e, ...a) => cb(...a)),
  onSessionInfo:      (cb) => ipcRenderer.on('session-info',      (_e, ...a) => cb(...a)),
  onTrialLogged:      (cb) => ipcRenderer.on('trial-logged',      (_e, ...a) => cb(...a)),
});
