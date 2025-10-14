################################ Librerias #############################################################################
from pylsl import StreamInlet, resolve_stream
import numpy as np
from pylsl import StreamInfo, StreamOutlet
from scipy import signal
from filterpy.kalman import KalmanFilter
################################ Librerias #############################################################################


######################## LSL INPUT EEG #################################################################################
print("looking for an EEG stream...")
streams = resolve_stream('name', 'AURA_Power')
inlet = StreamInlet(streams[0])
######################## LSL INPUT EEG #################################################################################


###################### Funciones #######################################################################################
class Notch:
    def __init__(self, cutoff=50, Q=30, fs=250):
        '''
        Funcion para aplicar filtro pasa alto en una señal de un canal
        :param cutoff: Corte de frecuencia, banda que se eliminara
        :param Q: 
        :param fs: Frecuencia de muestreo de la senal
        '''
        self.cutoff = cutoff
        self.Q = Q
        nyq = 0.5 * fs
        w0 = cutoff / nyq
        self.b, self.a = signal.iirnotch(w0, Q)

    def process(self, data, axis=0):
        return signal.filtfilt(self.b, self.a, data, axis)

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    [b, a] = signal.butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = signal.filtfilt(b, a, data, axis=0)
    return y
###################### Funciones #######################################################################################


########################### Variables ##################################################################################
SCALE_FACTOR_EEG = (4500000)/24/(2**23-1) #uV/count
fs = 250
cutNotch = 50
cutBP = (1.0, 50.0)
orden = 5
nCanales = 3
raweeg = np.zeros((1, nCanales))
oldSample = None
ready = False
calibsamples = round(fs / 2)
zif = []
for i in range(nCanales):
    zif.append([[], []])
buffVisual = np.zeros((1, nCanales))
buffStora = np.zeros((1, nCanales))
# Configuración inicial del Filtro de Kalman
kf = KalmanFilter(dim_x=nCanales, dim_z=nCanales)  # Ajusta según el número de canales
kf.x = np.zeros(nCanales)  # Estado inicial
kf.F = np.eye(nCanales)  # Matriz de transición de estado
kf.H = np.eye(nCanales)  # Matriz de observación
kf.P *= 1000.     # Covarianza de estado inicial
kf.R = 0.5        # Covarianza de observación (ruido)
kf.Q = 0.5        # Covarianza del proceso (ruido del modelo)
########################### Variables ##################################################################################


###################### LSL OUTPUT EEG ##################################################################################
info = StreamInfo('AURAFilteredEEG', 'EEG', nCanales, fs, 'float32', 'pythonFlt')
info_channels = info.desc().append_child("channels")
for c in range(nCanales):
    info_channels.append_child("channel").append_child_value("label", "ch" + str(c + 1))
info.desc().append_child_value("sampling_frequency", str(fs))
outlet = StreamOutlet(info)

# Crear un segundo StreamOutlet para la señal filtrada por Kalman
info_kalman = StreamInfo('AURAKalmanFilteredEEG', 'EEG', nCanales, fs, 'float32', 'pythonKlmFlt')
outlet_kalman = StreamOutlet(info_kalman)
###################### LSL OUTPUT EEG ##################################################################################


###################### Ejecucion #######################################################################################
print("Iniciando captura...")
while True:
    sample0, timestamp = inlet.pull_sample()
    sample= [float(x)*SCALE_FACTOR_EEG for x in sample0[:3]]

    try:
        rawdata = np.reshape(np.asarray(sample[:nCanales]), (1, nCanales))
    except:
        sample = oldSample
        rawdata = np.reshape(np.asarray(sample[:nCanales]), (1, nCanales))

    if np.shape(raweeg)[0] < calibsamples:
        if np.mean(raweeg) == 0:
            raweeg[0] = rawdata
        else:
            raweeg = np.concatenate((raweeg, rawdata))
            if np.shape(raweeg)[0] == calibsamples:
                print("Recoleccion completada, iniciando filtrado...")
    else:
        raweeg = np.concatenate((raweeg, rawdata))
        raweeg = raweeg[1:np.shape(raweeg)[0]]
        filterEEG = raweeg.copy()
        filterEEG = raweeg.copy().transpose()
        for i in range(len(filterEEG)):
            fltNotch = Notch(cutNotch, fs=fs)
            filterEEG[i] = fltNotch.process(filterEEG[i])
            filterEEG[i] = butter_bandpass_filter(filterEEG[i], lowcut=cutBP[0], highcut=cutBP[1], fs=fs, order=orden)
        # Enviar la señal después de los filtros notch y pasa-banda
        outlet.push_sample(filterEEG[:, -1])
        # Aplicar el Filtro de Kalman
        kf.predict()  # Predicción del estado
        latest_measurement = filterEEG[:, -1]  # Obtener la última medición de todos los canales
        kf.update(latest_measurement.reshape(-1, 1))  # Actualización con las mediciones

        # Utilizar el estado filtrado por el Kalman
        kalman_filtered_sample = kf.x

        # Enviar la señal después del filtro de Kalman
        outlet_kalman.push_sample(kalman_filtered_sample)

    oldSample = sample
###################################################### Ejecucion #######################################################