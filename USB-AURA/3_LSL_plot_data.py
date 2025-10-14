from pylsl import StreamInlet, resolve_stream, resolve_bypred
import numpy as np
import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import psutil
import collections

SCALE_FACTOR_EEG = (4500000)/24/(2**23-1) #uV/count

print("looking for an EEG stream...")
#streams = resolve_stream('name', 'AURA')
#SCALE_FACTOR_EEG = (4500000)/24/(2**23-1) #if raw data

streams = resolve_stream('name', 'AURAFilteredEEG')
SCALE_FACTOR_EEG = 1 #if data is already filtered

#streams = resolve_stream('name', 'AURA_Power')

plot_channel=2 # channels 0 to 7

inlet = StreamInlet(streams[0])

def update_my_plot(datapoint):
    sample, timestamp = inlet.pull_sample()
    channel_data = [float(x)*SCALE_FACTOR_EEG for x in sample]
    print(channel_data)

    eeg_stream.popleft()
    eeg_stream.append(channel_data[plot_channel])
    # clear axis
    ax.cla()
    # plot stream
    ax.plot(eeg_stream)
    ax.scatter(len(eeg_stream)-1, eeg_stream[-1])
    ax.text(len(eeg_stream)-1, eeg_stream[-1]+2, "{}%".format(eeg_stream[-1]))
    #ax.set_ylim(-1,1)
    ax.set_ylim(min(eeg_stream),max(eeg_stream))




eeg_stream=collections.deque(np.zeros(100))
# define and adjust figure
fig = plt.figure(figsize=(6,12), facecolor='#DEDEDE')
ax = plt.subplot(111)
ax.set_facecolor('#DEDEDE')


ani = FuncAnimation(fig, update_my_plot,interval=100)
plt.show()
