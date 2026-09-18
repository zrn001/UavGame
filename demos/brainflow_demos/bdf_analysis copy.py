import time
import mne
import numpy as np
from scipy import signal, fftpack
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from metabci.brainda.algorithms.decomposition import FBTRCA
from metabci.brainda.algorithms.decomposition.base import generate_filterbank
from matplotlib import pyplot as plt
import pickle

mne.set_log_level(False)

def segment_epoch(data, event, tmin=0.0, tmax=1.0, fs=250):
    """ Segment epochs. """
    # Segment epochs
    n_event, n_chan, n_tpoint = event.shape[0], data.shape[0], int(fs*tmax-fs*tmin)
    epochs = np.zeros((n_event, n_chan, n_tpoint))
    for idx_event in range(n_event):
        idx_t0 = event[idx_event, 0] + int(fs*tmin)
        idx_t1 = event[idx_event, 0] + int(fs*tmax)
        epochs[idx_event] = data[:, idx_t0:idx_t1]
    return epochs


# In[1]: Config
event_map = {str(e):e for e in range(1, 9)}
fs = 250
tmax = 5.14
tmin = 0.14
interval = [tmin, tmax]
n_chan = 8
freqs = [8, 9, 10, 11, 12, 13, 14, 15]
n_target = len(freqs)
file_path = 'E:\\2-ChinaMobile\\MetaBCI-zhz\\data\\Tju-testOffline\\1201\\2313text.bdf'


raw = mne.io.read_raw_bdf(file_path, preload=True)
data = raw.get_data()

epoch_raw = []
labels = []

event, _ = mne.events_from_annotations(raw, event_id=event_map)
labels.append(event[:, -1]-1)

# Raw for classification
data_raw = data.copy()
epoch_raw.append(segment_epoch(data_raw.copy(), event, interval[0], interval[1], fs))

epoch_raw = np.concatenate(epoch_raw, 0)
labels = np.concatenate(labels, 0)
print("labels:",labels)
print(epoch_raw.shape)
print("epoch:", epoch_raw[:,0,:])
data_npy = np.load('E:\\2-ChinaMobile\\MetaBCI-zhz\\data\\Tju-testOffline\\1201_11.npy')
print("npy:", data_npy[:,0,:])


srate = 250
f0 = 50
Q = 30
b_notch, a_notch = signal.iirnotch(f0, Q, fs=srate)

# epoch_raw = signal.filtfilt(b_notch, a_notch, epoch_raw)
# data_npy = signal.filtfilt(b_notch, a_notch, data_npy)
a=1
# plt.plot(data_npy[2, 0, :], label='online')
# plt.plot(epoch_raw[2, 0, :], label='offline')
# plt.legend()
# plt.show()


pkl_path = 'E:\\2-ChinaMobile\\MetaBCI-zhz\\all_data.pkl'
f = open(pkl_path,'rb')
data = pickle.load(f)
data = np.concatenate(data, axis=0)


event_data = np.nonzero(data[..., -1])
# if event_data
print(data)

