import csv
from pylsl import StreamInlet, resolve_stream

# Nombre del archivo CSV
nombre_archivo_csv = "datos_psd2.csv"

# Resolver y conectar al stream LSL 'AURAPSD'
print("Buscando el stream 'AURAPSD'...")
streams = resolve_stream('type', 'PSD')
inlet = StreamInlet(streams[0])
print("Conectado al stream.")

# Configurar el archivo CSV
with open(nombre_archivo_csv, mode='w', newline='') as archivo_csv:
    escritor_csv = csv.writer(archivo_csv)
    
    # Escribir los nombres de las columnas (ajustar según el número de electrodos y bandas)
    nombres_columnas = ['Delta1', 'Theta1', 'Alpha1', 'Beta1', 'Gamma1',
                        'Delta2', 'Theta2', 'Alpha2', 'Beta2', 'Gamma2',
                        'Delta3', 'Theta3', 'Alpha3', 'Beta3', 'Gamma3']
    escritor_csv.writerow(nombres_columnas)

    # Leer datos del stream y guardar en CSV
    try:
        while True:
            sample, timestamp = inlet.pull_sample()
            escritor_csv.writerow(sample)
            print(f"Datos guardados: {sample}")
    except KeyboardInterrupt:
        print("Guardado finalizado.")
