################################ Librerias #############################################################################

from pylsl import StreamInlet, resolve_stream
import numpy as np
from pylsl import StreamInfo, StreamOutlet
from scipy import signal

################################################################ Librerias #############################################


######################## LSL INPUT EEG #################################################################################

print("looking for an EEG stream...")
streams = resolve_stream('name', 'AURA')
inlet = StreamInlet(streams[0])


############################################################### LSL INPUT EEG ##########################################


###################### Funciones #######################################################################################

class Notch:
    def __init__(self, cutoff=50, Q=30, fs=256):
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


############################################################### Funciones ##############################################


########################### Variables ##################################################################################

fs = 250
cutNotch = 50
cutBP = (1.0, 50.0)
orden = 5
nCanales = 8
raweeg = np.zeros((1, nCanales))
oldSample = None
ready = False
calibsamples = round(fs / 2)
zif = []
for i in range(nCanales):
    zif.append([[], []])

buffVisual = np.zeros((1, nCanales))
buffStora = np.zeros((1, nCanales))

####################################################### Variables ######################################################


###################### LSL OUTPUT EEG ##################################################################################

info = StreamInfo('AURAFilteredEEG', 'EEG', nCanales, fs, 'float32', 'pythonFlt')
info_channels = info.desc().append_child("channels")
for c in range(nCanales):
    info_channels.append_child("channel").append_child_value("label", "ch" + str(c + 1))
info.desc().append_child_value("sampling_frequency", str(fs))
outlet = StreamOutlet(info)

####################################################### LSL OUTPUT EEG #################################################


###################### Ejecucion #######################################################################################

print("Iniciando captura...")
while True:
    sample, timestamp = inlet.pull_sample()

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
        filterEEG = filterEEG.transpose()
        for i in range(len(filterEEG)):
            fltNotch = Notch(cutNotch, fs=fs)
            filterEEG[i] = fltNotch.process(filterEEG[i])
            filterEEG[i] = butter_bandpass_filter(filterEEG[i], lowcut=cutBP[0], highcut=cutBP[1], fs=fs,
                                                  order=orden)
        print(filterEEG[:, len(filterEEG[0]) - 1])
        outlet.push_sample(filterEEG[:, len(filterEEG[0]) - 1])

    oldSample = sample

###################################################### Ejecucion #######################################################