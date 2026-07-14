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


inlet = StreamInlet(streams[0])

def update_my_plot(datapoint):
    sample, timestamp = inlet.pull_sample()
    channel_data = [float(x)*SCALE_FACTOR_EEG for x in sample]
    print(channel_data)

# ********** Plot 1 *****************************
    plot_channel=0
    eeg_stream.popleft()
    eeg_stream.append(channel_data[plot_channel])
    # clear axis
    ax.cla()
    ax.set_title('Channel 1')
    # plot stream
    ax.plot(eeg_stream)
    ax.scatter(len(eeg_stream)-1, eeg_stream[-1])
    ax.text(len(eeg_stream)-1, eeg_stream[-1]+2, "{}%".format(eeg_stream[-1]))
    ax.set_ylim(min(eeg_stream),max(eeg_stream))

# ********** Plot 2 *****************************
    plot_channel=1
    eeg_stream1.popleft()
    eeg_stream1.append(channel_data[plot_channel])
    # clear axis
    ax1.cla()
    ax1.set_title('Channel 2')
    # plot stream
    ax1.plot(eeg_stream1)
    ax1.scatter(len(eeg_stream1)-1, eeg_stream1[-1])
    ax1.text(len(eeg_stream1)-1, eeg_stream1[-1]+2, "{}%".format(eeg_stream1[-1]))
    ax1.set_ylim(min(eeg_stream1),max(eeg_stream1))

# ********** Plot 2 *****************************
    plot_channel=2
    eeg_stream2.popleft()
    eeg_stream2.append(channel_data[plot_channel])
    # clear axis
    ax2.cla()
    ax2.set_title('Channel 3')
    # plot stream
    ax2.plot(eeg_stream2)
    ax2.scatter(len(eeg_stream2)-1, eeg_stream2[-1])
    ax2.text(len(eeg_stream2)-1, eeg_stream2[-1]+2, "{}%".format(eeg_stream2[-1]))
    ax2.set_ylim(min(eeg_stream2),max(eeg_stream2))


eeg_stream=collections.deque(np.zeros(100))
eeg_stream1=collections.deque(np.zeros(100))
eeg_stream2=collections.deque(np.zeros(100))
# define and adjust figure
fig = plt.figure(figsize=(12,12), facecolor='#DEDEDE')
ax = plt.subplot(311)
ax1 = plt.subplot(312)
ax2 = plt.subplot(313)

#ax.set_facecolor('#DEDEDE')


ani = FuncAnimation(fig, update_my_plot,interval=100)
plt.show()
