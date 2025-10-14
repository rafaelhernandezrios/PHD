from pylsl import StreamInlet, resolve_stream, resolve_bypred
import numpy as np

SCALE_FACTOR_EEG = (4500000)/24/(2**23-1) #uV/count


print("looking for an EEG stream...")
streams = resolve_stream('name', 'AURA')
#streams = resolve_stream('type', 'EEG')
#streams = resolve_stream('name', 'AURA_Power')



inlet = StreamInlet(streams[0])
while True:
    # get a new sample (you can also omit the timestamp part if you're not
    # interested in it)
    sample, timestamp = inlet.pull_sample()
    channel_data = [float(x)*SCALE_FACTOR_EEG for x in sample]
    print(timestamp, channel_data)
    