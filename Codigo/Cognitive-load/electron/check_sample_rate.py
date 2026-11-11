#!/usr/bin/env python3
"""
Verifica que la AURA este entregando datos a la tasa esperada (250 Hz).

Dos modos:
  1) Live:
       venv/bin/python electron/check_sample_rate.py
     Se conecta al stream LSL "AURA" y mide la tasa REAL de llegada de
     muestras (usando el reloj LSL, no el reloj de llegada al SO) durante
     ~10 s, y la compara contra lo que el propio stream declara como
     nominal_srate() en su metadata.

  2) Offline:
       venv/bin/python electron/check_sample_rate.py ruta/a/eeg_raw.csv
     Calcula la tasa efectiva a partir de la columna de timestamps LSL de
     una sesion ya grabada (funciona con eeg_raw.csv del pipeline nuevo,
     columna "timestamp_lsl", o con los eeg_data_*.csv viejos, columna
     "timestamp"). No necesita la diadema conectada.

Sale con codigo 0 si la tasa medida cae dentro de +-5% de 250 Hz, 1 si no,
2 si no pudo medir nada.
"""
import sys
import time
import csv
import statistics

EXPECTED_HZ = 250.0
TOLERANCE = 0.05  # 5%


def check_offline(path):
    ts_col = None
    values = []
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        for cand in ("timestamp_lsl", "timestamp"):
            if cand in header:
                ts_col = header.index(cand)
                break
        if ts_col is None:
            print(f"No encontre columna de timestamp ('timestamp_lsl' o 'timestamp') en {header}")
            sys.exit(2)
        for row in reader:
            try:
                values.append(float(row[ts_col]))
            except (ValueError, IndexError):
                continue

    if len(values) < 10:
        print(f"Muy pocas filas validas ({len(values)}) para calcular una tasa.")
        sys.exit(2)

    span = values[-1] - values[0]
    n = len(values) - 1
    if span <= 0:
        print("Los timestamps no avanzan (span <= 0); no puedo calcular la tasa.")
        sys.exit(2)
    effective_hz = n / span

    diffs = sorted(b - a for a, b in zip(values, values[1:]))
    median_dt = statistics.median(diffs)

    print(f"Archivo: {path}")
    print(f"Muestras: {len(values)}  |  Duracion (reloj LSL): {span:.2f} s")
    print(f"Tasa efectiva (muestras / span de timestamps): {effective_hz:.2f} Hz")
    print(f"dt mediano entre muestras: {median_dt * 1000:.3f} ms  (-> {1.0 / median_dt:.2f} Hz)")
    print(f"dt min/max: {diffs[0] * 1000:.3f} ms / {diffs[-1] * 1000:.3f} ms")
    _verdict(effective_hz)


def check_live(duration=10.0):
    try:
        from pylsl import resolve_byprop, StreamInlet
    except ImportError:
        print("pylsl no esta instalado en este interprete.")
        print("Corre este script con el venv del proyecto, ej:")
        print("  venv/bin/python electron/check_sample_rate.py")
        sys.exit(2)

    print("Buscando stream LSL 'AURA' (timeout 5 s)...")
    streams = resolve_byprop("name", "AURA", timeout=5.0)
    if not streams:
        print("No encontre ningun stream 'AURA'.")
        print("Confirma que AuraSetup / el dispositivo esten conectados y transmitiendo.")
        sys.exit(2)

    info = streams[0]
    declared = info.nominal_srate()
    print(f"Stream encontrado: {info.name()} | canales: {info.channel_count()} | "
          f"nominal_srate() declarado: {declared} Hz")

    inlet = StreamInlet(info, max_buflen=360)
    print(f"Midiendo tasa real durante {duration:.0f} s (reloj LSL, no reloj de llegada al SO)...")

    first_ts = last_ts = None
    n = 0
    t_start = time.time()
    while time.time() - t_start < duration:
        chunk, timestamps = inlet.pull_chunk(timeout=1.0)
        if timestamps:
            if first_ts is None:
                first_ts = timestamps[0]
            last_ts = timestamps[-1]
            n += len(timestamps)

    if not first_ts or not last_ts or last_ts <= first_ts or n < 2:
        print("No llegaron suficientes muestras para medir. ¿Esta la diadema realmente transmitiendo?")
        sys.exit(2)

    effective_hz = (n - 1) / (last_ts - first_ts)
    print(f"Muestras recibidas: {n}  |  Ventana medida (reloj LSL): {last_ts - first_ts:.2f} s")
    print(f"Tasa declarada por el stream (nominal_srate): {declared:.2f} Hz")
    print(f"Tasa REAL medida (timestamps LSL): {effective_hz:.2f} Hz")
    if declared and abs(effective_hz - declared) / declared > TOLERANCE:
        print("AVISO: la tasa real medida NO coincide con lo que el stream declara (nominal_srate).")
        print("       Esto apunta a que el dispositivo/firmware esta entregando datos a otra")
        print("       frecuencia (p.ej. modo crudo del ADC) sin que el metadata LSL se haya")
        print("       actualizado. Revisa la configuracion de sample rate en AuraSetup / el")
        print("       dispositivo antes de grabar una sesion real.")
    _verdict(effective_hz)


def _verdict(effective_hz):
    low, high = EXPECTED_HZ * (1 - TOLERANCE), EXPECTED_HZ * (1 + TOLERANCE)
    if low <= effective_hz <= high:
        print(f"OK: {effective_hz:.2f} Hz esta dentro de +-{TOLERANCE * 100:.0f}% de los {EXPECTED_HZ:.0f} Hz esperados.")
        sys.exit(0)
    print(f"FALLA: {effective_hz:.2f} Hz esta fuera de +-{TOLERANCE * 100:.0f}% de los {EXPECTED_HZ:.0f} Hz esperados.")
    print("No arranques 'Setup Session' / no confies en estos datos hasta resolverlo.")
    sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        check_offline(sys.argv[1])
    else:
        check_live()
