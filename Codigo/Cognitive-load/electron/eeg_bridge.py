"""Electron <-> Python bridge for Cognitive-load."""

import sys
import os
import json
import threading
import time
from datetime import datetime

# ── path setup ────────────────────────────────────────────────────────
# Project root is one level up from electron/
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Optional backend import
try:
    from core.signal_worker import SignalWorker
    from pylsl import resolve_streams, resolve_byprop, StreamInlet
    HAS_BACKEND = True
except ImportError as e:
    HAS_BACKEND = False
    _import_error = str(e)


def emit(event: str, data: dict):
    """Write one JSON line to stdout for Electron to consume."""
    payload = json.dumps({"event": event, "data": data}, ensure_ascii=True)
    sys.stdout.write(payload + "\n")
    sys.stdout.flush()


LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(LOG_DIR, exist_ok=True)
BRIDGE_LOG_PATH = os.path.join(LOG_DIR, "eeg_bridge.log")


def bridge_log(message: str, data=None):
    """Append debug line to bridge log file."""
    try:
        line = {
            "ts": datetime.now().isoformat(),
            "message": message,
            "data": data,
        }
        with open(BRIDGE_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(line, ensure_ascii=True) + "\n")
    except Exception:
        pass


class EEGBridge:
    def __init__(self):
        self.signal_worker = None
        self.connected = False
        self.is_logging = False
        self.data_log = []
        self._log_buffer = []
        self._log_counter = 0
        self._subsampling_factor = 5   # 250 Hz → 50 Hz
        self._log_buffer_size = 50
        self.current_phase = "idle"
        self.current_user = None
        self.user_folder = None
        self._phase_labels = {
            'idle': 'idle',
            'setup': 'setup',
            'baseline_eyes_open': 'baseline_eyes_open',
            'baseline_eyes_closed': 'baseline_eyes_closed',
            'baseline_completed': 'baseline_completed',
            'low_load': 'low_cognitive_load',
            'low_load_completed': 'low_load_completed',
            'high_load': 'high_cognitive_load',
            'high_load_completed': 'high_load_completed',
            'analysis': 'analysis',
            'completed': 'completed',
            'ecological_paradigm': 'ecological_paradigm',
        }
        self.ecological_recording = False
        self.ecological_modality = None  # eco_gamepad | eco_haptic | eco_keyboard
        self._allowed_eco_modalities = frozenset({"eco_gamepad", "eco_haptic", "eco_keyboard"})

        # Background workers
        self._ratio_thread = None
        self._ratio_running = False
        self._phase_timer_thread = None
        self._phase_timer_running = False
        self.phase_durations = {
            "baseline_eyes_open": 90,
            "baseline_eyes_closed": 90,
            "low_load": 180,
            "high_load": 180,
        }

    def _ensure_logging_session(self):
        """Guarantee a valid session so Save works even without explicit setup."""
        if not self.current_user:
            self.current_user = datetime.now().strftime("session_%Y%m%d_%H%M%S")
            self.user_folder = f"data_{self.current_user}"
            os.makedirs(self.user_folder, exist_ok=True)
        elif not self.user_folder:
            self.user_folder = f"data_{self.current_user}"
            os.makedirs(self.user_folder, exist_ok=True)
        self.is_logging = True

    def connect(self):
        bridge_log("connect_called")
        if not HAS_BACKEND:
            bridge_log("backend_import_error", {"error": _import_error})
            emit("connection_status", {"connected": False, "message": f"Import error: {_import_error}"})
            return

        if self.signal_worker and self.signal_worker.isRunning():
            bridge_log("connect_ignored_already_running")
            self.connected = True
            emit("connection_status", {"connected": True, "message": ""})
            return

        try:
            bridge_log("creating_signal_worker")
            self.signal_worker = SignalWorker(sample_rate=250, n_channels=8, buffer_duration=2.0)
            self.signal_worker.connection_status.connect(self._on_connection_status)
            # No Qt event loop in this process: use direct callbacks (see SignalWorker).
            self.signal_worker._electron_bridge_data = self._electron_data_direct
            self.signal_worker._electron_bridge_plot = self._electron_plot_direct

            chosen = self._pick_best_stream()
            if chosen is not None:
                bridge_log(
                    "stream_selected",
                    {
                        "name": str(chosen.name()),
                        "type": str(chosen.type()),
                        "source_id": str(chosen.source_id()),
                        "channels": int(chosen.channel_count()),
                        "sample_rate": float(chosen.nominal_srate()),
                    },
                )
                self.signal_worker.inlet = StreamInlet(chosen)
                info = chosen
                emit(
                    "device_info",
                    {
                        "lsl_found": True,
                        "inlet_active": True,
                        "stream_name": str(info.name()),
                        "sample_rate": float(info.nominal_srate()),
                        "channels": int(info.channel_count()),
                        "visible_streams": self._scan_streams(),
                    },
                )
                self.connected = True
                self.signal_worker.start()
                self._ensure_logging_session()
                bridge_log("signal_worker_started", {"is_running": bool(self.signal_worker.isRunning())})
                emit("connection_status", {"connected": True, "message": ""})
                emit("phase_durations", self.phase_durations)
                self._start_ratio_polling()
            else:
                bridge_log("no_stream_selected")
                self.connected = False
                emit("device_info", {"lsl_found": False, "inlet_active": False, "visible_streams": self._scan_streams()})
                emit("connection_status", {"connected": False, "message": "No LSL stream found."})
        except Exception as exc:
            bridge_log("connect_exception", {"error_type": type(exc).__name__, "error": str(exc)})
            self.connected = False
            emit("device_info", {"lsl_found": False, "inlet_active": False, "visible_streams": self._scan_streams()})
            emit("connection_status", {"connected": False, "message": str(exc)})

    def disconnect(self):
        bridge_log("disconnect_called")
        self._ratio_running = False
        self._phase_timer_running = False
        if (
            self._ratio_thread
            and self._ratio_thread.is_alive()
            and self._ratio_thread is not threading.current_thread()
        ):
            self._ratio_thread.join(timeout=0.3)
        if (
            self._phase_timer_thread
            and self._phase_timer_thread.is_alive()
            and self._phase_timer_thread is not threading.current_thread()
        ):
            self._phase_timer_thread.join(timeout=0.3)
        if self.signal_worker and self.signal_worker.isRunning():
            self.signal_worker.stop()
            self.signal_worker.wait()
        self.signal_worker = None
        self.connected = False
        self.ecological_recording = False
        self.ecological_modality = None
        emit("ecological_state", {"active": False, "modality": None})
        emit("device_info", {"lsl_found": False, "inlet_active": False, "visible_streams": []})
        emit("connection_status", {"connected": False, "message": ""})

    def _on_connection_status(self, connected, message):
        bridge_log("connection_status_signal", {"connected": bool(connected), "message": str(message)})
        emit("connection_status", {"connected": bool(connected), "message": str(message)})

    def _scan_streams(self):
        if not HAS_BACKEND:
            return []
        result = []
        try:
            streams = resolve_streams(wait_time=1.0)
            bridge_log("scan_streams_result", {"count": len(streams)})
            for s in streams[:12]:
                try:
                    result.append(
                        {
                            "name": str(s.name()),
                            "type": str(s.type()),
                            "source_id": str(s.source_id()),
                            "channels": int(s.channel_count()),
                            "sample_rate": float(s.nominal_srate()),
                        }
                    )
                except Exception:
                    pass
        except Exception:
            bridge_log("scan_streams_exception")
            pass
        return result

    def _pick_best_stream(self):
        """Pick best LSL stream for AURA acquisition."""
        # 1) Strict match first (legacy expected stream name)
        streams = resolve_byprop("name", "AURA", timeout=1.5)
        bridge_log("resolve_byprop_name_AURA", {"count": len(streams)})
        # 2) Fallback: broad discovery and scoring
        if not streams:
            streams = resolve_streams(wait_time=1.5)
            bridge_log("resolve_streams_fallback", {"count": len(streams)})
        if not streams:
            return None

        scored = []
        for s in streams:
            try:
                name = (s.name() or "").lower()
                stype = (s.type() or "").lower()
                sid = (s.source_id() or "").lower()
                ch = int(s.channel_count())
                sr = float(s.nominal_srate())
            except Exception:
                continue

            score = 0
            if "aura" in name:
                score += 100
            if "aura" in sid:
                score += 60
            if stype == "eeg":
                score += 25
            if ch == 8:
                score += 30
            if 200 <= sr <= 300:
                score += 20
            if "power" in name:
                score -= 40

            scored.append((score, s))

        if not scored:
            return None
        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[0][1]
        bridge_log(
            "top_scored_stream",
            {
                "score": scored[0][0],
                "name": str(top.name()),
                "type": str(top.type()),
                "source_id": str(top.source_id()),
            },
        )
        return scored[0][1]

    def scan_lsl(self):
        visible = self._scan_streams()
        aura_like = any("aura" in (s.get("name", "").lower()) for s in visible)
        inlet_active = bool(self.signal_worker and self.signal_worker.isRunning()) or bool(self.connected)
        emit(
            "device_info",
            {
                "lsl_found": bool(aura_like or len(visible) > 0),
                "inlet_active": inlet_active,
                "visible_streams": visible,
            },
        )

    def start_setup(self, user: str):
        if not user:
            emit("save_error", {"error": "User is required."})
            return
        self.current_user = user
        self.user_folder = f"data_{user}"
        os.makedirs(self.user_folder, exist_ok=True)
        self.current_phase = "setup"
        self.is_logging = True
        self._log_counter = 0
        emit("phase_changed", {"phase": "setup", "message": ""})
        emit("timer_update", {"time": self._fmt_time(0)})

    def start_baseline(self):
        self.current_phase = "baseline_eyes_open"
        emit("phase_changed", {"phase": "baseline_eyes_open", "message": ""})
        self._start_timer(int(self.phase_durations["baseline_eyes_open"]), "baseline_eyes_open")

    def start_low_load(self):
        self.current_phase = "low_load"
        emit("phase_changed", {"phase": "low_load", "message": ""})
        self._start_timer(int(self.phase_durations["low_load"]), "low_load")

    def start_high_load(self):
        self.current_phase = "high_load"
        emit("phase_changed", {"phase": "high_load", "message": ""})
        self._start_timer(int(self.phase_durations["high_load"]), "high_load")

    def set_phase_durations(self, payload: dict):
        def _clamp(v, default):
            try:
                x = int(v)
                return max(5, min(3600, x))
            except Exception:
                return default

        self.phase_durations["baseline_eyes_open"] = _clamp(
            payload.get("baseline_eyes_open"), self.phase_durations["baseline_eyes_open"]
        )
        self.phase_durations["baseline_eyes_closed"] = _clamp(
            payload.get("baseline_eyes_closed"), self.phase_durations["baseline_eyes_closed"]
        )
        self.phase_durations["low_load"] = _clamp(
            payload.get("low_load"), self.phase_durations["low_load"]
        )
        self.phase_durations["high_load"] = _clamp(
            payload.get("high_load"), self.phase_durations["high_load"]
        )
        emit("phase_durations", self.phase_durations)

    def _start_timer(self, seconds: int, phase: str):
        self._phase_timer_running = False
        if (
            self._phase_timer_thread
            and self._phase_timer_thread.is_alive()
            and self._phase_timer_thread is not threading.current_thread()
        ):
            self._phase_timer_thread.join(timeout=0.2)
        self._phase_timer_running = True
        self._phase_timer_thread = threading.Thread(
            target=self._phase_timer_loop, args=(seconds, phase), daemon=True
        )
        self._phase_timer_thread.start()

    def _phase_timer_loop(self, seconds: int, phase: str):
        remaining = int(seconds)
        emit("timer_update", {"time": self._fmt_time(remaining)})
        while self._phase_timer_running and remaining > 0:
            time.sleep(1)
            remaining -= 1
            emit("timer_update", {"time": self._fmt_time(remaining)})

        if not self._phase_timer_running:
            return

        if phase == "baseline_eyes_open":
            self.current_phase = "baseline_eyes_closed"
            emit("phase_changed", {"phase": "baseline_eyes_closed", "message": ""})
            self._start_timer(int(self.phase_durations["baseline_eyes_closed"]), "baseline_eyes_closed")
        elif phase == "baseline_eyes_closed":
            self.current_phase = "baseline_completed"
            emit("phase_changed", {"phase": "baseline_completed", "message": ""})
        elif phase == "low_load":
            self.current_phase = "low_load_completed"
            emit("phase_changed", {"phase": "low_load_completed", "message": ""})
        elif phase == "high_load":
            self.current_phase = "analysis"
            emit("phase_changed", {"phase": "analysis", "message": ""})

    @staticmethod
    def _fmt_time(seconds: int) -> str:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes:02d}:{secs:02d}"

    def _log_row_phase_and_eco(self):
        """While ecological recording is on, rows are tagged as ecological_paradigm + modality."""
        if self.ecological_recording and self.ecological_modality:
            return "ecological_paradigm", self.ecological_modality
        return self.current_phase, None

    def _label_for_record(self, phase: str, eco_mod):
        if phase == "ecological_paradigm" and eco_mod:
            return f"ecological_paradigm mode={eco_mod}"
        return self._phase_labels.get(phase, phase)

    def _on_data_ready(self, data, timestamp):
        if not self.is_logging:
            return
        self._log_counter += 1
        if self._log_counter % self._subsampling_factor != 0:
            return
        ph, eco = self._log_row_phase_and_eco()
        self._log_buffer.append((data, float(timestamp), ph, eco))
        if len(self._log_buffer) >= self._log_buffer_size:
            self._flush_buffer()
        emit("sample_count", {"count": int(len(self.data_log) + len(self._log_buffer))})

    def _electron_data_direct(self, data, timestamp):
        """Called from SignalWorker thread; avoids Qt queued delivery (no event loop in this process)."""
        self._on_data_ready(data, timestamp)

    def _electron_plot_direct(self, raw_arr, filt_arr, timestamp):
        """Called from SignalWorker thread with paired raw + filtered samples."""
        raw = [float(x) for x in raw_arr]
        filt = [float(x) for x in filt_arr]
        emit("plot_sample", {"raw": raw, "filtered": filt, "timestamp": float(timestamp)})

    def _flush_buffer(self):
        for row in self._log_buffer:
            data, ts, phase, eco_mod = row
            eco_val = eco_mod if eco_mod else ""
            record = {
                'timestamp': ts,
                'phase': phase,
                'label': self._label_for_record(phase, eco_mod),
                'ecological_modality': eco_val,
                'channel_0': float(data[0]) if len(data) > 0 else float('nan'),
                'channel_1': float(data[1]) if len(data) > 1 else float('nan'),
                'channel_2': float(data[2]) if len(data) > 2 else float('nan'),
                'channel_3': float(data[3]) if len(data) > 3 else float('nan'),
                'channel_4': float(data[4]) if len(data) > 4 else float('nan'),
                'channel_5': float(data[5]) if len(data) > 5 else float('nan'),
                'channel_6': float(data[6]) if len(data) > 6 else float('nan'),
                'channel_7': float(data[7]) if len(data) > 7 else float('nan'),
            }
            self.data_log.append(record)
        self._log_buffer.clear()

    def ecological_start(self, modality: str):
        m = (modality or "").strip()
        if m not in self._allowed_eco_modalities:
            emit("ecological_error", {"error": f"Invalid modality: use eco_gamepad, eco_haptic, or eco_keyboard."})
            return
        if not self.connected or not (self.signal_worker and self.signal_worker.isRunning()):
            emit("ecological_error", {"error": "Connect AURA first."})
            return
        if not self.current_user:
            emit("ecological_error", {"error": "Complete Setup Session first."})
            return
        self.ecological_recording = True
        self.ecological_modality = m
        emit("ecological_state", {"active": True, "modality": m})

    def ecological_stop(self):
        self.ecological_recording = False
        self.ecological_modality = None
        emit("ecological_state", {"active": False, "modality": None})

    def _start_ratio_polling(self):
        self._ratio_running = False
        if (
            self._ratio_thread
            and self._ratio_thread.is_alive()
            and self._ratio_thread is not threading.current_thread()
        ):
            self._ratio_thread.join(timeout=0.2)
        self._ratio_running = True
        self._ratio_thread = threading.Thread(target=self._ratio_loop, daemon=True)
        self._ratio_thread.start()

    def _ratio_loop(self):
        while self._ratio_running:
            time.sleep(2.0)
            if self.signal_worker and self.signal_worker.isRunning():
                result = self.signal_worker.get_cognitive_load_ratio()
                if result is not None:
                    ratio, theta, alpha = result
                    emit("ratio_update", {
                        "ratio": float(ratio),
                        "theta": float(theta),
                        "alpha": float(alpha),
                    })

    def save_data(self):
        try:
            import pandas as pd
        except ImportError:
            emit("save_error", {"error": "Missing dependency: pandas. Install project requirements first."})
            return
        if self._log_buffer:
            self._flush_buffer()
        if not self.current_user:
            self._ensure_logging_session()
        if not self.data_log and self.signal_worker:
            # Fallback snapshot from current ring buffer if user saves early.
            try:
                window = self.signal_worker.ring_buffer.get_window(self.signal_worker.buffer_samples)
                ph, eco = self._log_row_phase_and_eco()
                eco_val = eco if eco else ""
                for row in window:
                    vals = [float(x) for x in row[:8]]
                    record = {
                        "timestamp": float(time.time()),
                        "phase": ph,
                        "label": self._label_for_record(ph, eco),
                        "ecological_modality": eco_val,
                        "channel_0": vals[0] if len(vals) > 0 else float("nan"),
                        "channel_1": vals[1] if len(vals) > 1 else float("nan"),
                        "channel_2": vals[2] if len(vals) > 2 else float("nan"),
                        "channel_3": vals[3] if len(vals) > 3 else float("nan"),
                        "channel_4": vals[4] if len(vals) > 4 else float("nan"),
                        "channel_5": vals[5] if len(vals) > 5 else float("nan"),
                        "channel_6": vals[6] if len(vals) > 6 else float("nan"),
                        "channel_7": vals[7] if len(vals) > 7 else float("nan"),
                    }
                    self.data_log.append(record)
            except Exception:
                pass
        if not self.data_log:
            emit("save_error", {"error": "No data to save yet. Keep connected for a few seconds and try again."})
            return
        try:
            df = pd.DataFrame(self.data_log)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"eeg_data_{ts}.csv"
            filepath = os.path.join(self.user_folder, filename)
            df.to_csv(filepath, index=False)
            emit("save_done", {"rows": len(df), "filepath": filepath})
        except Exception as exc:
            emit("save_error", {"error": str(exc)})


# ─────────────────────────────────────────────────────────────────────
# STDIN COMMAND LOOP
# ─────────────────────────────────────────────────────────────────────
def main():
    bridge_log("bridge_main_started", {"python": sys.executable})
    bridge = EEGBridge()
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue

        cmd = msg.get("cmd", "")
        bridge_log("command_received", {"cmd": cmd})
        if cmd == "connect":
            bridge.connect()
        elif cmd == "disconnect":
            bridge.disconnect()
        elif cmd == "start_setup":
            bridge.start_setup(msg.get("user", "unknown"))
        elif cmd == "start_baseline":
            bridge.start_baseline()
        elif cmd == "start_low_load":
            bridge.start_low_load()
        elif cmd == "start_high_load":
            bridge.start_high_load()
        elif cmd == "save_data":
            bridge.save_data()
        elif cmd == "shutdown":
            bridge.disconnect()
            break
        elif cmd == "scan_lsl":
            bridge.scan_lsl()
        elif cmd == "set_phase_durations":
            bridge.set_phase_durations(msg)
        elif cmd == "ecological_start":
            bridge.ecological_start(msg.get("modality", ""))
        elif cmd == "ecological_stop":
            bridge.ecological_stop()


if __name__ == "__main__":
    main()
