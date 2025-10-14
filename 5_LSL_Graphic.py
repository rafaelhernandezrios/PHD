import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np
from pylsl import StreamInlet, resolve_stream
import matplotlib.pyplot as plt
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
# Resolver y conectar al stream LSL para EEG sin filtro de Kalman
print("Buscando el stream 'AURAFilteredEEG'...")
streams = resolve_stream('name', 'AURAFilteredEEG')
inlet = StreamInlet(streams[0])

# Resolver y conectar al stream LSL para EEG con filtro de Kalman
print("Buscando el stream 'AURAKalmanFilteredEEG'...")
streams_kalman = resolve_stream('name', 'AURAKalmanFilteredEEG')
inlet_kalman = StreamInlet(streams_kalman[0])

# Configuración inicial
fs = 250  # Frecuencia de muestreo
nCanales = 3  # Número de canales

# Crear ventana de Tkinter
root = tk.Tk()
root.title("Visualización EEG en Tiempo Real")

# Crear dos gráficas de matplotlib
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
canvas = FigureCanvasTkAgg(fig, master=root)
canvas_widget = canvas.get_tk_widget()
canvas_widget.pack(fill=tk.BOTH, expand=True)

# Configurar las gráficas
ax1.set_title("Señal sin Kalman")
ax2.set_title("Señal con Kalman")
for ax in [ax1, ax2]:
    ax.set_xlabel("Tiempo (s)")
    ax.set_ylabel("Amplitud (uV)")
    ax.set_ylim(-1,1)  # Establecer límites del eje Y
# Inicializar buffers para almacenar datos
buffer_length = 1 * fs  # 5 segundos de datos
buffer_sin_kalman = np.zeros((buffer_length, nCanales))
buffer_con_kalman = np.zeros((buffer_length, nCanales))

# Función para actualizar gráficas
def update_graph():
    global buffer_sin_kalman, buffer_con_kalman

    # Obtener nuevos datos de los streams LSL
    sample, timestamp = inlet.pull_sample()
    sample_kalman, timestamp_kalman = inlet_kalman.pull_sample()

    # Actualizar buffers
    buffer_sin_kalman = np.roll(buffer_sin_kalman, -1, axis=0)
    buffer_con_kalman = np.roll(buffer_con_kalman, -1, axis=0)
    buffer_sin_kalman[-1, :] = sample
    buffer_con_kalman[-1, :] = sample_kalman

    # Actualizar gráficas
    for i in range(nCanales):
        ax1.lines[i].set_ydata(buffer_sin_kalman[:, i])
        ax2.lines[i].set_ydata(buffer_con_kalman[:, i])
    
    canvas.draw()
    root.after(100, update_graph)  # Actualizar cada 100 ms

# Configurar líneas iniciales en las gráficas
for i in range(nCanales):
    ax1.plot(np.arange(buffer_length) / fs, buffer_sin_kalman[:, i], label=f'Canal {i+1}')
    ax2.plot(np.arange(buffer_length) / fs, buffer_con_kalman[:, i], label=f'Canal {i+1}')
ax1.legend()
ax2.legend()

# Iniciar la actualización de las gráficas
root.after(1, update_graph)

# Iniciar la interfaz gráfica
root.mainloop()
