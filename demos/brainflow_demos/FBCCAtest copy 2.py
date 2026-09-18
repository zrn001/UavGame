import matplotlib.pyplot as plt
import numpy as np
from metabci.brainda.datasets import Wang2016
from metabci.brainda.paradigms import SSVEP
from metabci.brainda.algorithms.utils.model_selection import (
    set_random_seeds,
    generate_kfold_indices,
    match_kfold_indices)
from metabci.brainda.algorithms.decomposition import FBTRCA, FBSCCA
from metabci.brainda.algorithms.decomposition.base import generate_filterbank, generate_cca_references
from mne.io import read_raw_bdf, read_raw_cnt
import mne
from metabci.brainda.algorithms.utils.model_selection import (
    EnhancedLeaveOneGroupOut)
from scipy.signal import filtfilt, lfilter


def label_encoder(y, labels):
    new_y = y.copy()
    for i, label in enumerate(labels):
        ix = (y == label)
        new_y[ix] = i
    return new_y


def read_bdf_file(run_files, trials, nlabels, stimlen, chs):

    fs = 1000
    delay = 0.14
    labels = list(range(1, nlabels + 1))
    for run_file in run_files:
        Xs, ys = [], []
        raw = read_raw_bdf(run_file, preload=True, verbose=False)
        # raw.notch_filter(np.arange(50, 251, 50), n_jobs=1)
        events = mne.events_from_annotations(raw, event_id=lambda x: int(x), verbose=False)[0]
        ch_picks = mne.pick_channels(raw.ch_names, chs, ordered=True)
        epochs = mne.Epochs(raw, events, event_id=labels, tmin=delay, tmax=delay + stimlen, baseline=None, picks=ch_picks,
                            preload=True, verbose=False)

        for label in labels:
            X = epochs[str(label)].get_data()[..., 1:]
            y = np.ones(len(X))*label
            Xs.append(X)
            ys.append(y)
        Xs_all = np.concatenate(Xs, axis=0)
        # Xs_all = Xs_all - np.mean(Xs_all, axis=2, keepdims=True)
        ys_all = np.concatenate(ys, axis=0)
        ys_all = label_encoder(ys_all, labels)
    return Xs_all, ys_all


# '''列建模'''
# # # If everything is fine, you will get the accuracy about 0.9417.
srate=1000
stimlen = 1
sample_point = int(srate*stimlen)

pick_chs = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'E4', 'F4', 'C13', 'C14', 'C15']

freq_list = [8, 12, 8.8, 12.8, 9.6, 13.6, 10.4, 14.4, 11.2, 15.2]
Yf = generate_cca_references(freq_list, srate=1000, T=1,n_harmonics = 5)

wp = [[6, 88], [14, 88], [22, 88], [30, 88], [38, 88]
]
ws = [[4, 90], [12, 90], [20, 90], [28, 90], [36, 90]
]
filterweights = np.arange(1, 6)**(-1.25) + 0.25
filterbank = generate_filterbank(wp, ws, 1000)


'''在线'''
# online_filepath = 'D:\\ZFEEG\\app\\data\\20240828zaz_dry_new\\2step\\offline'
online_filepath = 'C:\\Users\\Administrator\\Desktop\\3-China-Mobile\\小姜师姐FBSCCA'
n_cnt_online = range(1, 2)
online_run_files = ['{:s}\\{:d}.bdf'.format(online_filepath, run) for run in n_cnt_online]
n_label = 10
n_trial = 10

'''online'''
online_stimlen = 1

X_simu, y_simu = read_bdf_file(online_run_files, n_trial, n_label, online_stimlen, pick_chs)
print(X_simu)
print('read data successful')
kfold_accs = []
model = FBSCCA(filterbank=filterbank, n_components=1, filterweights=filterweights, n_jobs=-1)
model.fit(X_simu, y_simu, Yf=Yf)
p_label = model.predict(X_simu)
kfold_accs.append(np.mean(p_label == y_simu))
print(np.mean(kfold_accs))

