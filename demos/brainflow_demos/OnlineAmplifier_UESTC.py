# -*- coding: utf-8 -*-
"""
SSAVEP Feedback on NeuroScan.

"""
import time
import numpy as np

import mne
from mne.filter import resample
from pylsl import StreamInfo, StreamOutlet
from metabci.brainflow.amplifiers import NeuroScan, New_Uestc, Marker
from metabci.brainflow.workers import ProcessWorker
from metabci.brainda.algorithms.decomposition.base import (
    generate_filterbank, generate_cca_references)
from metabci.brainda.algorithms.utils.model_selection import (
    EnhancedLeaveOneGroupOut)
from metabci.brainda.algorithms.decomposition import FBTDCA,FBTRCA
from metabci.brainda.utils import upper_ch_names
from mne.io import read_raw_cnt
from sklearn.base import BaseEstimator, ClassifierMixin


def label_encoder(y, labels):
    new_y = y.copy()
    for i, label in enumerate(labels):
        ix = (y == label)
        new_y[ix] = i
    return new_y


class MaxClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self):
        pass

    def fit(self, X, y):
        pass

    def predict(self, X):
        X = X.reshape((-1, X.shape[-1]))
        y = np.argmax(X, axis=-1)
        return y


def read_data(run_files, chs, interval, labels, srate=2000):
    ##########################################################################################
    # Load each channel data
    Xs, ys = [], []
    ch_picks = [int(c.split('_')[-1]) for c in chs]
    for run_file in run_files:
        data_raw = []
        for idx_chan in range(len(chs)):
            with open("{}/{}.txt".format(run_file, chs[idx_chan]), "r") as f:
                data_chan = f.read()
            data_chan = list(map(int, data_chan.split(" ")[:-1]))
            data_chan = np.array(data_chan, dtype=np.float64)*5*1000*1000/(8388608)/24
            data_raw.append(data_chan)
            del data_chan
        data_raw = np.stack(data_raw, axis=0)
        
        # Load marker
        with open("{}//marker.txt".format(run_file), "r") as f:
            marker = f.read()
        marker = list(map(int, marker.split(" ")[:-1]))
        marker = np.array(marker)
        idx_marker = np.where(marker!=0)[0]

        # Segment epochs
        n_marker, n_chan, n_tpoint = len(idx_marker), data_raw.shape[0], int(srate*interval[1]-srate*interval[0])
        X = np.zeros((n_marker, n_chan, n_tpoint))
        for idx in range(n_marker):
            idx_t0 = idx_marker[idx] + int(srate*interval[0])
            idx_t1 = idx_marker[idx] + int(srate*interval[1])
            X[idx] = data_raw[:, idx_t0:idx_t1]
        y = marker[idx_marker] - 1
        
        Xs.append(X)
        ys.append(y)
    Xs = np.concatenate(Xs, axis=0)
    ys = np.concatenate(ys, axis=0)
        
    return Xs, ys, ch_picks
    ##########################################################################################

def train_model(X, y, srate=1000):
    y = np.reshape(y, (-1))
    X = resample(X, up=256, down=srate)

    wp = [
        [6, 88], [14, 88], [22, 88], [30, 88], [38, 88]
    ]
    ws = [
        [4, 90], [12, 90], [20, 90], [28, 90], [36, 90]
    ]

    filterweights = np.arange(1, 6)**(-1.25) + 0.25
    filterbank = generate_filterbank(wp, ws, 256)
    X = X - np.mean(X, axis=-1, keepdims=True)
    X = X / np.std(X, axis=(-1, -2), keepdims=True)

    #freqs = np.arange(8, 16, 0.4)
    #Yf = generate_cca_references(freqs, srate=256, T=0.5, n_harmonics=5)
    Yf = None
    # model = FBTDCA(filterbank, padding_len=3, n_components=4,
    #                 filterweights=np.array(filterweights))
    model = FBTRCA(filterbank, n_components=4,
                   filterweights=np.array(filterweights))

    model = model.fit(X, y ,Yf)

    return model


def model_predict(X, srate=1000, model=None):
    X = np.reshape(X, (-1, X.shape[-2], X.shape[-1]))
    X = resample(X, up=256, down=srate)
    X = X - np.mean(X, axis=-1, keepdims=True)
    X = X / np.std(X, axis=(-1, -2), keepdims=True)
    p_labels = model.predict(X)
    return p_labels


def offline_validation(X, y, srate=1000):
    y = np.reshape(y, (-1))
    spliter = EnhancedLeaveOneGroupOut(return_validate=False)

    kfold_accs = []
    for train_ind, test_ind in spliter.split(X, y=y):
        X_train, y_train = np.copy(X[train_ind]), np.copy(y[train_ind])
        X_test, y_test = np.copy(X[test_ind]), np.copy(y[test_ind])

        model = train_model(X_train, y_train, srate=srate)
        p_labels = model_predict(X_test, srate=srate, model=model)
        kfold_accs.append(np.mean(p_labels == y_test))
    return np.mean(kfold_accs)


class FeedbackWorker(ProcessWorker):
    def __init__(self, run_files, pick_chs, stim_interval, stim_labels,
                 srate, lsl_source_id, timeout, worker_name):
        self.run_files = run_files
        self.pick_chs = pick_chs
        self.stim_interval = stim_interval
        self.stim_labels = stim_labels
        self.srate = srate
        self.lsl_source_id = lsl_source_id
        super().__init__(timeout=timeout, name=worker_name)

    def pre(self):
        X, y, ch_ind = read_data(run_files=self.run_files,
                                  chs=self.pick_chs,
                                  interval=self.stim_interval,
                                  labels=self.stim_labels,
                                  srate=self.srate)
        
        print("Loding train data successfully")
        # Compute offline acc
        acc = offline_validation(X, y, srate=self.srate)
        print("Current Model accuracy:{:.2f}".format(acc))
        self.estimator = train_model(X, y, srate=self.srate)
        # self.ch_ind = ch_ind

        info = StreamInfo(
            name='meta_feedback',
            type='Markers',
            channel_count=1,
            nominal_srate=0,
            channel_format='int32',
            source_id=self.lsl_source_id)
        self.outlet = StreamOutlet(info)
        print('Waiting connection...')
        while not self._exit:
            if self.outlet.wait_for_consumers(1e-3):
                break
        print('Connected')

    def consume(self, data):
        data = np.array(data, dtype=np.float64).T
        data = data[self.ch_ind]
        p_labels = model_predict(data, srate=self.srate, model=self.estimator)
        p_labels = np.array([int(p_labels + 1)])
        # p_labels = p_labels.tolist()
        p_labels = list(p_labels)
        print('predict_id_paradigm', p_labels)
        if self.outlet.have_consumers():
            self.outlet.push_sample(p_labels)

    def post(self):
        pass


if __name__ == '__main__':
    ##########################################################################################
    # Sample rate EEG amplifier

    # In[1]: Config
    srate = 250
    event_map = {str(e):e for e in range(1, 9)}
    # tmax = 2.14
    # tmin = 0.14
    n_chan = 8
    freqs = [8, 9, 10, 11, 12, 13, 14, 15]
    n_target = len(freqs)
    run_files = ['./Data/Tju-testOffline/1128/1.bdf', './Data/Tju-testOffline/1128/2.bdf']

    # srate = 2000
    # # Data epoch duration, 0.14s visual delay was taken account
    stim_interval = [0.14, 2.14]
    # # Label types
    stim_labels = list(range(1, 9))
    # cnts = 1
    # # Data path
    # # filepath = "data\\train\\sub1"
    # filepath = "C:\\Users\\Administrator\\Desktop\\MetaBCI-zhz\\data\\Uestc-testOffline\\#Device1"
    # runs = list(range(1, cnts+1))
    # run_files = ['{:s}/run{}'.format(filepath, run) for run in runs]
    pick_chs = ['Channel_1', 'Channel_2', 'Channel_3', 'Channel_4', 'Channel_5', 'Channel_6', 'Channel_7', 'Channel_8', ]
    ##########################################################################################

    lsl_source_id = 'mobile-china'
    feedback_worker_name = 'feedback_worker'

    worker = FeedbackWorker(
        run_files=run_files,
        pick_chs=pick_chs,
        stim_interval=stim_interval,
        stim_labels=stim_labels, srate=srate,
        lsl_source_id=lsl_source_id,
        timeout=5e-2,
        worker_name=feedback_worker_name)
    marker = Marker(interval=stim_interval, srate=srate,
                    events=stim_labels)

    worker.pre()
    # Set New_Uestc parameters
    ns = New_Uestc(
        device_address=('127.0.0.1', 9687),
        srate=srate,
        num_chans=8)

    # Start tcp connection with ns
    ns.connect_tcp()
    # Start acquire data from ns
    ns.start_acq()

    # Register worker for online data processing56
    ns.register_worker(feedback_worker_name, worker, marker)
    # Start online data processing
    ns.up_worker(feedback_worker_name)
    time.sleep(0.5)

    # Start slicing data and passing data to worker
    ns.start_trans()

    # input('press any key to close\n')
    # ns.down_worker('feedback_worker')
    # time.sleep(1)

    # # Stop online data retriving of ns
    # ns.stop_trans()
    # ns.stop_acq()
    # ns.close_connection()
    # ns.clear()
    # print('bye')
