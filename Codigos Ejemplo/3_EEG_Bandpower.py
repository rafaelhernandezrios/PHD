#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This file receives an LSL stream of eight EEG channels with filtered EEG data

The output is a 40 channel stream contaning normalized bandpower values
for previous electrodes in the following form:
Alpha CH1
Alpha CH2
.
.
Alpha CH8
Beta CH1~CH8
Gamma CH1~CH8
Delta CH1~CH8
Theta CH1~CH8    
"""

import pylsl
import numpy as np
import scipy
from datetime import datetime

# Brainwave band value defininitions
alpha = [8.0, 12.0]
beta = [12.0, 30.0]
gamma = [30.0, 60.0]
delta = [0.0, 4.0]  # 0.1
theta = [4.0, 8.0]

alpha_power = np.empty(8)
beta_power = np.empty(8)
gamma_power = np.empty(8)
delta_power = np.empty(8)
theta_power = np.empty(8)

# Number of channels
n_channels = 8
epoch_size = 250  # 250
window_size = 50
# Device sampling frequency
sampling_frequency = 250

# Nyquist criteria
dt = 1 / sampling_frequency  # Sampling period
fn = 1 / (2 * dt)  # Nyquist frequency

raw_data = np.zeros(8)
EEGepochdata = np.zeros((8, epoch_size))
TempEpoc = np.zeros((8, window_size))
TSepochdata = np.zeros(epoch_size)
TSTempEpoc = np.zeros(window_size)


# Low-pass filter function it seems it is not used on computing function)
def lowpass(x):
    # Low-pass filter declarations and definition
    fp = 0.45  # Passband frequency [Hz]
    fst = 30  # Stopband frequency [Hz]
    gpass = 1  # Bandpass ripple [dB]
    gstop = 60  # Stopband attenuation [dB]
    # Filter definitions
    Wp = fp / fn
    Wst = fst / fn
    # Filter implementation
    N, Wn = scipy.signal.buttord(Wp, Wst, gpass, gstop)
    b1, a1 = scipy.signal.butter(N, Wn, 'low')
    return scipy.signal.filtfilt(b1, a1, x)


def nextpow2(n):
    n_log = np.log2(n)
    N_log = np.ceil(n_log)
    return int(N_log)


def myfft(x, fs):
    m = len(x)
    n = 2 ** nextpow2(m)
    y = scipy.fft(x, n)
    f = np.arange(n) * fs / n
    power = y * np.conj(y) / n
    return f, np.real(power)


# Bandpower calculation snippet
def bandpower(x, fs):
    # f, Pxx = signal.periodogram(x, fs, nfft = 2 ** nextpow2(len(x)), return_onesided = True)
    f, Pxx = myfft(x, fs)
    return f, Pxx


def SplitDataToFreqBand(f, power, fmin, fmax):
    lowidx = scipy.argmax(f >= fmin)
    highidx = scipy.argmax(f > fmax) - 1
    band_power = np.mean(power[lowidx:highidx + 1])
    return band_power


# Bandpower normalization function
def normalization(band):
    a = np.min(np.abs(band), axis=0)
    b = np.max(np.abs(band), axis=0)
    if b - a == 0:
        band = 0
    else:
        band = (band - a) / (b - a)
    return band.tolist()


def ComputeBandPowerAll(EEGdata, fs):
    global alpha_power
    global beta_power
    global gamma_power
    global delta_power
    global theta_power

    for i in range(len(EEGdata)):
        f, power = bandpower(EEGdata[i], fs)
        alpha_power[i] = SplitDataToFreqBand(f, power, alpha[0], alpha[1])
        beta_power[i] = SplitDataToFreqBand(f, power, beta[0], beta[1])
        gamma_power[i] = SplitDataToFreqBand(f, power, gamma[0], gamma[1])
        delta_power[i] = SplitDataToFreqBand(f, power, delta[0], delta[1])
        theta_power[i] = SplitDataToFreqBand(f, power, theta[0], theta[1])
    alpha_power = normalization(alpha_power)
    beta_power = normalization(beta_power)
    gamma_power = normalization(gamma_power)
    delta_power = normalization(delta_power)
    theta_power = normalization(theta_power)
    normalized_bandpower = alpha_power + beta_power + gamma_power + delta_power + theta_power
    return normalized_bandpower


# select headset
while True:
    selection = input("Choose your headset \n1:Nautilus \n2:Polymate \n3:Aura \n=>")
    try:
        if int(selection) == 1:
            headset = "Nautilus"
            break
        elif int(selection) == 2:
            headset = "Polymate"
            break
        elif int(selection) == 3:
            headset = "Aura"
            break
        else:
            print("input 1 or 2")
    except:
        print("input 1 or 2")

# Stream Inlet for incoming LSL raw signal (Polymate or g.Nautilus)
if headset == "Aura":
    print("looking for an EEG stream...")
    streams = pylsl.resolve_stream('name', 'AURAFilteredEEG')
    # streams = pylsl.resolve_stream('name', 'AURA_Filtered')
    # streams = pylsl.resolve_stream('name', 'PythonFlt')
    raw_eeg = pylsl.StreamInlet(streams[0])
    print("Stream found...")

    # Normalized Bandpower stream
    info = pylsl.StreamInfo('EEG_Headset', 'EEG_BANDPOWER_X', (raw_eeg.channel_count * 5) + 1, epoch_size, 'float32',
                            'bandpower')

    info_channels = info.desc().append_child("channels")
    lblPB = ["a", "b", "g", "d", "t"]
    for pb in lblPB:
        for c in range(raw_eeg.channel_count):
            lblChn = pb + "Ch" + str(c + 1)
            info_channels.append_child("channel").append_child_value("label", lblChn)
    info.desc().append_child_value("epoch_size", str(epoch_size))
    info.desc().append_child_value("window_size", str(window_size))
    info.desc().append_child_value("sampling_frequency", str(sampling_frequency))
    bandpower_x = pylsl.StreamOutlet(info)

else:
    print("looking for an EEG stream...")
    streams = pylsl.resolve_stream('type', 'EEG')
    raw_eeg = pylsl.StreamInlet(streams[0])
    print("Stream found...")

    # Normalized Bandpower stream
    info = pylsl.StreamInfo('EEG_Headset', 'EEG_BANDPOWER_X', 41, epoch_size, 'float32', 'bandpower')
    info_channels = info.desc().append_child("channels")
    labels = ["aCH1", "aCH2", "aCH3", "aCH4", "aCH5", "aCH6", "aCH7", "aCH8",
              "bCH1", "bCH2", "bCH3", "bCH4", "bCH5", "bCH6", "bCH7", "bCH8",
              "gCH1", "gCH2", "gCH3", "gCH4", "gCH5", "gCH6", "gCH7", "gCH8",
              "dCH1", "dCH2", "dCH3", "dCH4", "dCH5", "dCH6", "dCH7", "dCH8",
              "tCH1", "tCH2", "tCH3", "tCH4", "tCH5", "tCH6", "tCH7", "tCH8"]
    for c in labels:
        info_channels.append_child("channel").append_child_value("label", c)
    info.desc().append_child_value("epoch_size", str(epoch_size))
    info.desc().append_child_value("window_size", str(window_size))
    info.desc().append_child_value("sampling_frequency", str(sampling_frequency))
    bandpower_x = pylsl.StreamOutlet(info)

# Program will execute until halted manually (Ctrl + C)
if headset == "Nautilus":
    electrode = 17
elif headset == "Polymate":
    electrode = 8
elif headset == "Aura":
    electrode = raw_eeg.channel_count #Toma el numero de canales que vienen en el LSL de Aura
try:
    count = 0
    window_cnt = 0
    none_c = 0
    while True:

        # Obtaining the most recent EEG value sample
        sample, timestamp = raw_eeg.pull_sample()
        # print(timestamp, sample)

        electrode_idx = 0
        if sample != None:
            for i in range(electrode):
                if headset == "Nautilus" and i != 0 and i != 1 and i != 5 and i != 7 and i != 9 and i < 13:
                    raw_data[electrode_idx] = sample[i]
                    electrode_idx += 1
                elif headset == "Polymate":
                    raw_data[i] = sample[i]
                elif headset == "Aura":
                    raw_data[i] = sample[i] #Todos los canales del LSL corresponden a un canal de AURA
            if headset == "Nautilus":
                head_ts = sample[16]
            elif headset == "Polymate":
                head_ts = sample[8]
            elif headset == "Aura":
                head_ts = datetime.now().timestamp() #Toma la hora actual del sistema para el timestamp

            if count < epoch_size:
                EEGepochdata[:, count] = raw_data
                TSepochdata[count] = head_ts
                count += 1
            else:
                if window_cnt < window_size:
                    TempEpoc[:, window_cnt] = raw_data
                    TSTempEpoc[window_cnt] = head_ts
                    window_cnt += 1
                else:
                    if window_size < epoch_size:
                        EEGepochdata[:, :epoch_size - window_size] = EEGepochdata[:, window_size:]
                        EEGepochdata[:, epoch_size - window_size:] = TempEpoc
                        TSepochdata[:epoch_size - window_size] = TSepochdata[window_size:]
                        TSepochdata[epoch_size - window_size:] = TSTempEpoc
                    else:
                        EEGepochdata = TempEpoc
                        TSepochdata = TSTempEpoc
                    normalized_bandpower = ComputeBandPowerAll(EEGepochdata, sampling_frequency)
                    normalized_bandpower.append(TSepochdata[0])
                    bandpower_x.push_sample(normalized_bandpower)
                    print(normalized_bandpower)
                    print(none_c)
                    print("\n")
                    window_cnt = 0

        else:
            print("None\n")
            none_c += 1

except KeyboardInterrupt:
    pass
