"""Electron <-> Python bridge for Cognitive-load.

2026-08 revision (ERP-ready acquisition):
  * raw AND filtered signal are written at the full 250 Hz (the previous build
    kept only the filtered signal, decimated 5x by plain sample dropping);
  * everything is streamed to disk incrementally, so a crash no longer costs
    the whole session;
  * stimulus events and Stroop trials are persisted with LSL timestamps, which
    is what makes stimulus-locked epoching possible;
  * a clock-synchronisation round trip maps the renderer's ``performance.now``
    onto the LSL clock;
  * acquisition health (effective Fs, dropped samples, flat/rail electrodes) is
    reported to the UI once per second.
"""

import sys
import os
import csv
import json
import threading
import time
from datetime import datetime

# ── path setup ────────────────────────────────────────────────────────
# Project root is one level up from electron/
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# All recordings land under <project>/data, never relative to the CWD that
# Electron happens to spawn us with.
DATA_ROOT = os.path.join(ROOT, "data")

# Optional backend import
try:
    from core.signal_worker import SignalWorker, make_inlet
    from pylsl import resolve_streams, resolve_byprop, local_clock
    HAS_BACKEND = True
except ImportError as e:
    HAS_BACKEND = False
    _import_error = str(e)

    def local_clock():
        return time.time()


N_CHANNELS = 8

_stdout_lock = threading.Lock()


def emit(event: str, data: dict):
    """Write one JSON line to stdout for Electron to consume."""
    payload = json.dumps({"event": event, "data": data}, ensure_ascii=True)
    with _stdout_lock:
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


class IncrementalCsv:
    """
    Append-only CSV writer that flushes on a size/time budget.

    The point is that the file on disk is always a valid, near-complete
    recording: if Electron dies mid-session, at most the last few hundred
    milliseconds are missing instead of the entire run.
    """

    def __init__(self, path, header, flush_rows=250, flush_seconds=2.0):
        self.path = path
        self.header = header
        self.flush_rows = flush_rows
        self.flush_seconds = flush_seconds
        self._pending = 0
        self._last_flush = time.time()
        self._rows_written = 0
        self._fh = open(path, "w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._fh)
        self._writer.writerow(header)
        self._fh.flush()

    def write_rows(self, rows):
        if not rows:
            return
        self._writer.writerows(rows)
        self._pending += len(rows)
        self._rows_written += len(rows)
        now = time.time()
        if self._pending >= self.flush_rows or (now - self._last_flush) >= self.flush_seconds:
            self.flush()

    def write_row(self, row):
        self.write_rows([row])

    def flush(self):
        try:
            self._fh.flush()
            os.fsync(self._fh.fileno())
        except Exception:
            pass
        self._pending = 0
        self._last_flush = time.time()

    @property
    def rows_written(self):
        return self._rows_written

    def close(self):
        try:
            self.flush()
            self._fh.close()
        except Exception:
            pass


class EEGBridge:
    def __init__(self):
        self.signal_worker = None
        self.connected = False
        self.is_logging = False
        self.current_phase = "idle"
        self.current_user = None
        self.session_dir = None

        self._writer_lock = threading.Lock()
        self.raw_csv = None
        self.filt_csv = None
        self.events_csv = None
        self.trials_csv = None
        self._sample_total = 0
        self._last_count_emit = 0.0
        self._trial_counter = 0

        # Clock alignment between the renderer (performance.now) and LSL.
        self.clock_offset = None
        self.clock_rtt_ms = None

        self.line_freq = 60.0  # Mexico. Set to 50.0 for 50 Hz mains.

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
        self._health_thread = None
        self._health_running = False
        self._phase_timer_thread = None
        self._phase_timer_running = False
        self.phase_durations = {
            "baseline_eyes_open": 90,
            "baseline_eyes_closed": 90,
            "low_load": 180,
            "high_load": 180,
        }

    # ------------------------------------------------------------------
    # Session / files
    # ------------------------------------------------------------------
    def _open_session(self, user=None):
        """Creates the session directory and the four output files."""
        self._close_session()

        if not user:
            user = datetime.now().strftime("session_%Y%m%d_%H%M%S")
        self.current_user = user
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = os.path.join(DATA_ROOT, f"data_{user}", f"{user}_{stamp}")
        os.makedirs(self.session_dir, exist_ok=True)

        chan_cols = [f"channel_{i}" for i in range(N_CHANNELS)]
        signal_header = ["timestamp_lsl", "phase", "label", "ecological_modality"] + chan_cols

        with self._writer_lock:
            self.raw_csv = IncrementalCsv(
                os.path.join(self.session_dir, "eeg_raw.csv"), signal_header
            )
            self.filt_csv = IncrementalCsv(
                os.path.join(self.session_dir, "eeg_filtered.csv"), signal_header
            )
            self.events_csv = IncrementalCsv(
                os.path.join(self.session_dir, "events.csv"),
                ["timestamp_lsl", "event", "phase", "detail"],
                flush_rows=1, flush_seconds=0.0,
            )
            self.trials_csv = IncrementalCsv(
                os.path.join(self.session_dir, "trials_stroop.csv"),
                [
                    "trial_index", "block", "word", "ink", "congruent",
                    "response", "correct", "rt_ms",
                    "onset_lsl", "response_lsl",
                    "onset_perf_ms", "response_perf_ms",
                    "clock_offset", "clock_rtt_ms",
                ],
                flush_rows=1, flush_seconds=0.0,
            )
            self._sample_total = 0
            self._trial_counter = 0

        self.is_logging = True
        self._write_session_meta()
        self.mark_event("session_start", {"user": user})
        emit("session_info", {"user": user, "session_dir": self.session_dir})
        bridge_log("session_opened", {"dir": self.session_dir})

    def _write_session_meta(self):
        if not self.session_dir:
            return
        meta = {
            "user": self.current_user,
            "created_at": datetime.now().isoformat(),
            "sample_rate_hz": 250,
            "n_channels": N_CHANNELS,
            "line_freq_hz": self.line_freq,
            "channel_map": ["Fp1", "Fp2", "F3", "Fz", "F4", "P3", "Pz", "P4"],
            "filter_raw": "none (ADC counts as delivered by AURA)",
            "filter_filtered": f"notch {self.line_freq} Hz (Q=30) -> butter bandpass 1-40 Hz order 4, causal",
            "clock_offset_perf_to_lsl": self.clock_offset,
            "clock_rtt_ms": self.clock_rtt_ms,
            "phase_durations": self.phase_durations,
            "note": (
                "events.csv is the authoritative source for phase and stimulus "
                "boundaries; the phase column in the signal files is chunk-level "
                "and can be off by up to one pull (~100 ms)."
            ),
        }
        try:
            with open(os.path.join(self.session_dir, "session.json"), "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)
        except Exception as exc:
            bridge_log("session_meta_error", {"error": str(exc)})

    def _close_session(self):
        with self._writer_lock:
            for w in (self.raw_csv, self.filt_csv, self.events_csv, self.trials_csv):
                if w:
                    w.close()
            self.raw_csv = self.filt_csv = self.events_csv = self.trials_csv = None
        self.is_logging = False

    def _ensure_logging_session(self):
        """Guarantee a valid session so data is never dropped on the floor."""
        if not self.session_dir:
            self._open_session(self.current_user)

    def mark_event(self, name, detail=None):
        """Writes a timestamped marker on the LSL clock."""
        if not self.events_csv:
            return None
        t = float(local_clock())
        row = [
            f"{t:.6f}",
            name,
            self.current_phase,
            json.dumps(detail or {}, ensure_ascii=True),
        ]
        with self._writer_lock:
            if self.events_csv:
                self.events_csv.write_row(row)
        return t

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------
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
            self.signal_worker = SignalWorker(
                sample_rate=250, n_channels=N_CHANNELS, buffer_duration=2.0,
                line_freq=self.line_freq,
            )
            self.signal_worker.connection_status.connect(self._on_connection_status)
            # No Qt event loop in this process: use direct callbacks.
            self.signal_worker._electron_bridge_chunk = self._on_chunk
            self.signal_worker._electron_bridge_plot = self._electron_plot_direct
            self.signal_worker._electron_bridge_status = self._on_worker_status

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
                self.signal_worker.inlet = make_inlet(chosen)
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
                # Deliberately NOT opening a session here. Connecting first and
                # running Setup afterwards used to create two folders per
                # participant: a throwaway "session_<timestamp>" with the few
                # seconds of montage fiddling, then the real one. The live
                # health monitor works without logging, so nothing is lost.
                bridge_log("signal_worker_started", {"is_running": bool(self.signal_worker.isRunning())})
                emit("connection_status", {"connected": True, "message": ""})
                emit("phase_durations", self.phase_durations)
                self._start_ratio_polling()
                self._start_health_polling()
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
        self.mark_event("session_end")
        self._ratio_running = False
        self._health_running = False
        self._phase_timer_running = False
        for th in (self._ratio_thread, self._health_thread, self._phase_timer_thread):
            if th and th.is_alive() and th is not threading.current_thread():
                th.join(timeout=0.3)
        if self.signal_worker and self.signal_worker.isRunning():
            self.signal_worker.stop()
            self.signal_worker.wait()
        self.signal_worker = None
        self.connected = False
        self.ecological_recording = False
        self.ecological_modality = None
        self._write_session_meta()
        self._close_session()
        emit("ecological_state", {"active": False, "modality": None})
        emit("device_info", {"lsl_found": False, "inlet_active": False, "visible_streams": []})
        emit("connection_status", {"connected": False, "message": ""})

    def _on_connection_status(self, connected, message):
        bridge_log("connection_status_signal", {"connected": bool(connected), "message": str(message)})
        emit("connection_status", {"connected": bool(connected), "message": str(message)})

    def _on_worker_status(self, payload):
        """Stream-loss / acquisition-error notices coming from the worker thread."""
        bridge_log("worker_status", payload)
        emit("stream_status", payload)
        if payload.get("stream_lost"):
            self.mark_event("stream_lost", payload)
        elif payload.get("stream_lost") is False:
            self.mark_event("stream_recovered", payload)

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

    # ------------------------------------------------------------------
    # Phases
    # ------------------------------------------------------------------
    def _set_phase(self, phase, message=""):
        self.current_phase = phase
        t = self.mark_event("phase_change", {"phase": phase})
        emit("phase_changed", {"phase": phase, "message": message, "timestamp_lsl": t})

    def start_setup(self, user: str):
        if not user:
            emit("save_error", {"error": "User is required."})
            return
        self._open_session(user)
        self._set_phase("setup")
        emit("timer_update", {"time": self._fmt_time(0)})

    def start_baseline(self):
        self._ensure_logging_session()
        self._set_phase("baseline_eyes_open")
        self._start_timer(int(self.phase_durations["baseline_eyes_open"]), "baseline_eyes_open")

    def start_low_load(self):
        self._ensure_logging_session()
        self._set_phase("low_load")
        self._start_timer(int(self.phase_durations["low_load"]), "low_load")

    def start_high_load(self):
        self._ensure_logging_session()
        self._set_phase("high_load")
        self._start_timer(int(self.phase_durations["high_load"]), "high_load")

    def set_phase_durations(self, payload: dict):
        def _clamp(v, default):
            try:
                x = int(v)
                return max(5, min(3600, x))
            except Exception:
                return default

        for key in ("baseline_eyes_open", "baseline_eyes_closed", "low_load", "high_load"):
            self.phase_durations[key] = _clamp(payload.get(key), self.phase_durations[key])
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
        # Deadline-based rather than accumulating sleep(1) errors.
        deadline = time.monotonic() + seconds
        remaining = int(seconds)
        emit("timer_update", {"time": self._fmt_time(remaining)})
        while self._phase_timer_running and remaining > 0:
            time.sleep(0.2)
            new_remaining = max(0, int(round(deadline - time.monotonic())))
            if new_remaining != remaining:
                remaining = new_remaining
                emit("timer_update", {"time": self._fmt_time(remaining)})

        if not self._phase_timer_running:
            return

        if phase == "baseline_eyes_open":
            self._set_phase("baseline_eyes_closed")
            self._start_timer(int(self.phase_durations["baseline_eyes_closed"]), "baseline_eyes_closed")
        elif phase == "baseline_eyes_closed":
            self._set_phase("baseline_completed")
        elif phase == "low_load":
            self._set_phase("low_load_completed")
        elif phase == "high_load":
            self._set_phase("analysis")

    @staticmethod
    def _fmt_time(seconds: int) -> str:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes:02d}:{secs:02d}"

    # ------------------------------------------------------------------
    # Data path
    # ------------------------------------------------------------------
    def _log_row_phase_and_eco(self):
        """While ecological recording is on, rows are tagged as ecological_paradigm + modality."""
        if self.ecological_recording and self.ecological_modality:
            return "ecological_paradigm", self.ecological_modality
        return self.current_phase, None

    def _label_for_record(self, phase: str, eco_mod):
        if phase == "ecological_paradigm" and eco_mod:
            return f"ecological_paradigm mode={eco_mod}"
        return self._phase_labels.get(phase, phase)

    def _on_chunk(self, raw_block, filt_block, timestamps):
        """
        Called from the SignalWorker thread with a whole block of samples.

        Both raw and filtered are written at the full sampling rate. No
        decimation: dropping 4 of every 5 samples without an anti-alias filter
        folded 38-42 Hz straight onto the 8-12 Hz alpha band that the
        cognitive-load index divides by.
        """
        if not self.is_logging or self.raw_csv is None:
            return

        phase, eco = self._log_row_phase_and_eco()
        label = self._label_for_record(phase, eco)
        eco_val = eco if eco else ""

        raw_rows = []
        filt_rows = []
        for i in range(len(raw_block)):
            ts = f"{float(timestamps[i]):.6f}"
            prefix = [ts, phase, label, eco_val]
            raw_rows.append(prefix + [f"{v:.6f}" for v in raw_block[i]])
            filt_rows.append(prefix + [f"{v:.6f}" for v in filt_block[i]])

        with self._writer_lock:
            if self.raw_csv is None:
                return
            self.raw_csv.write_rows(raw_rows)
            self.filt_csv.write_rows(filt_rows)
            self._sample_total += len(raw_rows)
            total = self._sample_total

        # Throttled: this used to fire once per logged sample (50 JSON lines
        # per second over stdout, on top of the plot stream).
        now = time.time()
        if now - self._last_count_emit >= 0.5:
            self._last_count_emit = now
            emit("sample_count", {"count": int(total)})

    def _electron_plot_direct(self, raw_arr, filt_arr, timestamp):
        """Called from SignalWorker thread with paired raw + filtered samples."""
        raw = [float(x) for x in raw_arr]
        filt = [float(x) for x in filt_arr]
        emit("plot_sample", {"raw": raw, "filtered": filt, "timestamp": float(timestamp)})

    # ------------------------------------------------------------------
    # Clock alignment (renderer performance.now -> LSL)
    # ------------------------------------------------------------------
    def clock_ping(self, msg):
        """
        Replies with the current LSL clock so the renderer can estimate the
        offset with Cristian's algorithm:

            offset = lsl_at_reply - (t0 + t1) / 2

        The renderer repeats this and keeps the estimate with the smallest
        round-trip time.
        """
        emit("clock_pong", {
            "id": msg.get("id"),
            "t0": msg.get("t0"),
            "lsl": float(local_clock()),
        })

    def set_clock_offset(self, msg):
        try:
            self.clock_offset = float(msg.get("offset"))
            self.clock_rtt_ms = float(msg.get("rtt_ms"))
        except (TypeError, ValueError):
            return
        self.mark_event("clock_sync", {
            "offset": self.clock_offset,
            "rtt_ms": self.clock_rtt_ms,
        })
        self._write_session_meta()
        emit("clock_sync_ok", {"offset": self.clock_offset, "rtt_ms": self.clock_rtt_ms})

    # ------------------------------------------------------------------
    # Behavioural events
    # ------------------------------------------------------------------
    def stroop_trial(self, msg):
        """
        Persists one Stroop trial. Previously this data existed only in the
        renderer's memory and was lost when the window closed.
        """
        self._ensure_logging_session()
        trial = msg.get("trial") or {}
        with self._writer_lock:
            self._trial_counter += 1
            idx = self._trial_counter
            writer = self.trials_csv

        def _num(key):
            v = trial.get(key)
            try:
                return f"{float(v):.6f}"
            except (TypeError, ValueError):
                return ""

        row = [
            idx,
            trial.get("block", ""),
            trial.get("word", ""),
            trial.get("ink", ""),
            int(bool(trial.get("congruent"))),
            trial.get("response", ""),
            int(bool(trial.get("correct"))),
            _num("rt_ms"),
            _num("onset_lsl"),
            _num("response_lsl"),
            _num("onset_perf_ms"),
            _num("response_perf_ms"),
            self.clock_offset if self.clock_offset is not None else "",
            self.clock_rtt_ms if self.clock_rtt_ms is not None else "",
        ]
        with self._writer_lock:
            if writer:
                writer.write_row(row)

        # Also drop markers into events.csv so epoching can be driven from a
        # single file.
        marker_detail = {
            "trial_index": idx,
            "word": trial.get("word"),
            "ink": trial.get("ink"),
            "congruent": bool(trial.get("congruent")),
        }
        self._write_marker(trial.get("onset_lsl"), "stroop_onset", marker_detail)
        self._write_marker(
            trial.get("response_lsl"), "stroop_response",
            dict(marker_detail, response=trial.get("response"),
                 correct=bool(trial.get("correct")), rt_ms=trial.get("rt_ms")),
        )
        emit("trial_logged", {"trial_index": idx})

    def _write_marker(self, t_lsl, name, detail):
        if t_lsl is None or not self.events_csv:
            return
        try:
            t = float(t_lsl)
        except (TypeError, ValueError):
            return
        row = [f"{t:.6f}", name, self.current_phase, json.dumps(detail, ensure_ascii=True)]
        with self._writer_lock:
            if self.events_csv:
                self.events_csv.write_row(row)

    def generic_mark(self, msg):
        """Marker requested by the UI (e.g. reading task start/stop)."""
        self._ensure_logging_session()
        name = msg.get("name") or "mark"
        t = self._write_marker(msg.get("t_lsl"), name, msg.get("detail") or {})
        if msg.get("t_lsl") is None:
            self.mark_event(name, msg.get("detail") or {})
        return t

    # ------------------------------------------------------------------
    # Ecological paradigm
    # ------------------------------------------------------------------
    def ecological_start(self, modality: str):
        m = (modality or "").strip()
        if m not in self._allowed_eco_modalities:
            emit("ecological_error", {"error": "Invalid modality: use eco_gamepad, eco_haptic, or eco_keyboard."})
            return
        if not self.connected or not (self.signal_worker and self.signal_worker.isRunning()):
            emit("ecological_error", {"error": "Connect AURA first."})
            return
        if not self.current_user:
            emit("ecological_error", {"error": "Complete Setup Session first."})
            return
        self.ecological_recording = True
        self.ecological_modality = m
        self.mark_event("ecological_start", {"modality": m})
        emit("ecological_state", {"active": True, "modality": m})

    def ecological_stop(self):
        self.mark_event("ecological_stop", {"modality": self.ecological_modality})
        self.ecological_recording = False
        self.ecological_modality = None
        emit("ecological_state", {"active": False, "modality": None})

    # ------------------------------------------------------------------
    # Background polling
    # ------------------------------------------------------------------
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

    def _start_health_polling(self):
        self._health_running = False
        if (
            self._health_thread
            and self._health_thread.is_alive()
            and self._health_thread is not threading.current_thread()
        ):
            self._health_thread.join(timeout=0.2)
        self._health_running = True
        self._health_thread = threading.Thread(target=self._health_loop, daemon=True)
        self._health_thread.start()

    def _health_loop(self):
        """
        Once a second, report effective sampling rate and electrode status.

        This is the check that would have caught the January sessions that
        silently recorded at ~13 Hz instead of 250 Hz, and the F4/P3 electrodes
        that sat railed at the ADC limit for entire runs.
        """
        warned_fs = False
        while self._health_running:
            time.sleep(1.0)
            if not (self.signal_worker and self.signal_worker.isRunning()):
                continue
            try:
                health = self.signal_worker.get_acquisition_health()
            except Exception as exc:
                bridge_log("health_error", {"error": str(exc)})
                continue

            eff = health.get("effective_fs", 0.0)
            nominal = health.get("nominal_fs", 250.0)
            bad = [c for c in health.get("channels", []) if c["status"] != "ok"]
            health["bad_channels"] = [c["index"] for c in bad]
            # Two-sided: a rate well ABOVE nominal means we are draining a
            # backlog or the stream is not what we think it is, and is just as
            # much a red flag as a rate below it.
            if health.get("warming_up"):
                health["fs_ok"] = True
            else:
                health["fs_ok"] = (0.9 * nominal) <= eff <= (1.15 * nominal)
            emit("acq_health", health)

            if not health["fs_ok"] and eff > 0 and not warned_fs:
                warned_fs = True
                self.mark_event("low_sampling_rate", {"effective_fs": eff, "nominal_fs": nominal})
            elif health["fs_ok"]:
                warned_fs = False

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    def save_data(self):
        """
        Data is already on disk; this just forces a flush and reports where it
        lives. Kept as a command so the existing UI button still works.
        """
        if not self.session_dir:
            emit("save_error", {"error": "No session open yet. Connect and run Setup first."})
            return
        with self._writer_lock:
            rows = self.raw_csv.rows_written if self.raw_csv else 0
            for w in (self.raw_csv, self.filt_csv, self.events_csv, self.trials_csv):
                if w:
                    w.flush()
        self._write_session_meta()
        if rows == 0:
            emit("save_error", {"error": "No samples recorded yet. Check the AURA connection."})
            return
        emit("save_done", {"rows": int(rows), "filepath": self.session_dir})


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
        # clock_ping fires many times per sync round; keep it out of the log.
        if cmd != "clock_ping":
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
        elif cmd == "clock_ping":
            bridge.clock_ping(msg)
        elif cmd == "clock_offset":
            bridge.set_clock_offset(msg)
        elif cmd == "stroop_trial":
            bridge.stroop_trial(msg)
        elif cmd == "mark":
            bridge.generic_mark(msg)


if __name__ == "__main__":
    main()
