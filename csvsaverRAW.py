import csv
from pylsl import StreamInlet, resolve_stream
from datetime import datetime

# Obtener el timestamp actual y formatearlo para el nombre del archivo
timestamp_actual = datetime.now().strftime("%Y%m%d_%H%M%S")
nombre_archivo_csv = f"datos_eeg_filtrados_{timestamp_actual}.csv"
nombre_archivo_csv2 = f"datos_eeg_filtrados_Kalman_{timestamp_actual}.csv"

# Resolver y conectar al stream LSL 'AURAFilteredEEG'
print("Buscando el stream 'AURAFilteredEEG'...")
streams = resolve_stream('name', 'AURAFilteredEEG')
inlet = StreamInlet(streams[0])
streams2 = resolve_stream('name', 'AURAKalmanFilteredEEG')
inlet2 = StreamInlet(streams2[0])
print("Conectado al stream.")

# Configurar el archivo CSV para AURAFilteredEEG
with open(nombre_archivo_csv, mode='w', newline='') as archivo_csv:
    escritor_csv = csv.writer(archivo_csv)
    nombres_columnas = ['Timestamp', 'Canal1', 'Canal2', 'Canal3']
    escritor_csv.writerow(nombres_columnas)

    # Configurar el archivo CSV para AURAKalmanFilteredEEG
    with open(nombre_archivo_csv2, mode='w', newline='') as archivo_csv2:
        escritor_csv2 = csv.writer(archivo_csv2)
        nombres_columnas2 = ['Timestamp', 'Canal1', 'Canal2', 'Canal3']
        escritor_csv2.writerow(nombres_columnas2)

        try:
            while True:
                # Leer datos del primer stream
                sample, timestamp = inlet.pull_sample()
                fila = [timestamp] + sample
                escritor_csv.writerow(fila)
                print(f"Datos guardados en AURAFilteredEEG: {fila}")

                # Leer datos del segundo stream
                sample2, timestamp2 = inlet2.pull_sample()
                fila2 = [timestamp2] + sample2
                escritor_csv2.writerow(fila2)
                print(f"Datos guardados en AURAKalmanFilteredEEG: {fila2}")

        except KeyboardInterrupt:
            print("Guardado finalizado.")
