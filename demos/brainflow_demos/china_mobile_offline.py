# -*- coding: utf-8 -*-
"""
SSAVEP Feedback on NeuroScan.

"""
import time
import numpy as np
import socket
import struct
from typing import Tuple

import mne
import scipy.io
from mne.filter import resample
from pylsl import StreamInfo, StreamOutlet
from scipy import signal, fftpack
from metabci.brainflow.amplifiers import NeuroScan, Marker,BaseAmplifier,RingBuffer
from metabci.brainflow.workers import ProcessWorker
from metabci.brainda.algorithms.decomposition.base import (
    generate_filterbank, generate_cca_references)
from metabci.brainda.algorithms.utils.model_selection import (
    EnhancedLeaveOneGroupOut)
from metabci.brainda.algorithms.decomposition import FBTDCA, FBTRCA, FBSCCA
from metabci.brainda.utils import upper_ch_names
from mne.io import read_raw_cnt
from sklearn.base import BaseEstimator, ClassifierMixin
import joblib
cre_list = []


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

def leave_one_block_out_test(data, label, model, n_block):
    """ Leave-one-block-out cross validation test. """
    result = []
    n_epoch = data.shape[0]
    n_epoch_block = int(data.shape[0]/n_block)
    # print("data.shape[0]:",data.shape[0])
    for idx_block in range(n_block):
        # Splid training and test set
        idx_test = np.arange(idx_block*n_epoch_block, (idx_block+1)*n_epoch_block)
        idx_train = np.setdiff1d(np.arange(n_epoch), idx_test)
        label_train = label[idx_train]
        label_test = label[idx_test]
        data_train = data[idx_train]
        data_test = data[idx_test] 
        # Fit model
        model.fit(data_train, label_train)
        # Get scores
        result.append(model.score(data_test, label_test))
    result = np.stack(result, axis=0)
    return result

def read_data(run_files, chs, interval, event_map,fs):
    epoch_raw = []
    labels = []
    for idx_bdf in range(len(run_files)):
        # Load data
        raw = mne.io.read_raw_bdf(run_files[idx_bdf], preload=True)
        # Interpolation for bad points
        data = raw.get_data()      
        
        # Event
        event, _ = mne.events_from_annotations(raw, event_id=event_map)
        labels.append(event[:, -1]-1)
        
        # Raw for classification
        data_raw = data.copy()
        epoch_raw.append(segment_epoch(data_raw.copy(), event, interval[0], interval[1], fs))
    epoch_raw = np.concatenate(epoch_raw, 0)
    np.save('./Data/Tju-testOffline/model_ssvep', epoch_raw)
    labels = np.concatenate(labels, 0)
    # print("labels:",labels)
    n_block3 = int(len(labels) / 8)
    # print("n_block3:",n_block3)

    return epoch_raw, labels, n_block3

def train_model():
    srate = 250
    f0 = 50
    Q = 30
    b_notch, a_notch = signal.iirnotch(f0, Q, fs=srate)

    # TRCA decoding model
    # wp = [(8,48), (16,48), (24,48)]
    # ws = [(6,50), (14,50), (22,50)]
    wp = [(5,90), (14,90), (22,90), (30,90), (38,90)]
    ws = [(3,92), (12,92), (20,92), (28,92), (36,92)]
    filterweights = np.array([(idx_filter+1)**(-1.25)+0.25 for idx_filter in range(len(wp))])
    filterbank = generate_filterbank(wp, ws, order=15, srate=srate, rp=0.5)
    model = FBTRCA(filterbank=filterbank, n_components=1, ensemble=True, filterweights=filterweights, n_jobs=-1)

    return model, b_notch, a_notch

def bandpass(sig, freq0, freq1, srate, axis=-1):
    wn1 = 2*freq0/srate
    wn2 = 2*freq1/srate
    b, a = signal.butter(4, [wn1, wn2], 'bandpass')
    sig_new = signal.filtfilt(b, a, sig, axis=axis)
    return sig_new

def model_predict(X, srate=250, model=None):
    srate = 250
    f0 = 50
    Q = 30
    b_notch, a_notch = signal.iirnotch(f0, Q, fs=srate)

    X = signal.filtfilt(b_notch, a_notch, X)

    X = np.reshape(X, (-1, X.shape[-2], X.shape[-1]))
    # print("X-shape:",X.shape)
    # print("X:",X)

    # print(X)
    # print(X.shape)
    p_labels = model.predict(X)    # EEG data, shape(n_trials, n_channels, n_samples).
    return p_labels


def offline_validation(X, y, n_block3, srate=250):
    t_list = np.round(np.arange(4.9, 5.1, 0.1), 1)
    acc = np.zeros((len(t_list)))
    model, b_notch, a_notch  = train_model()
    for idx_t, t in enumerate(t_list):
        t0 = time.time()
        # Notch filtering for suit time length
        epoch = signal.filtfilt(b_notch, a_notch, X[..., :int(t*srate)])
        # Leave-one-block-out cross validation
        result = leave_one_block_out_test(epoch, y, model, n_block=n_block3)
        # Accuracy
        acc[idx_t] = np.mean(result)
        print('Time:{} Acc:{:.3f}'.format(t, acc[idx_t]))

        model.fit(epoch, y)
        print('Training time:{:.4f}'.format(time.time()-t0))

        model_finally = model.fit(epoch, y)
    return model_finally


class NanoEEG(BaseAmplifier):

    def __init__(self,
                 device_address: Tuple[str, int] = ('127.0.0.1', 1895),
                 srate=250,
                 num_chans=8):
        super().__init__()
        self.device_address = device_address
        self.srate = srate
        self.num_chans = num_chans
        self.tcp_link = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.all_data = []
        self.data_count=0


    def connect_tcp(self):
        self.tcp_link.connect(self.device_address)

    def recv(self):
        data = None
        try:
            data = self.tcp_link.recv(9216)
           # print(data)
        except Exception as e:
            print("连接出错", e)
            self.tcp_link.close()
        finally:
            # 
            if data is not None:
                self.data_count +=1
                # print('finally1')
                data = self.__upack_data(data)
                # print(data)
                self.all_data.append(data)
                # if len(data) !=10 or len(data[0])!=9:
                #     print('Len: {}, {}'.format(len(data), len(data[0])))
                # if self.data_count == 1000:
                #     joblib.dump(self.all_data, "all_data.pkl")
                return data
            else:
                print("连接出错,此次请求数据失败，返回空数据")
                return []

    def __upack_data(self, data):
        # 一次十个采样点的数据 每个为int24 3个字节 (导联数+1)*10*3
        data = struct.unpack(f"{(self.num_chans + 1) * 10}i", data)
        data = np.array(data, dtype=np.int32)
        data = data.reshape((self.num_chans + 1, 10))
        data = np.transpose(data)

        return data.tolist()

    def start_trans(self):
        self.connect_tcp()
        print("连接成功")
        self.start()

    def stop_trans(self):
        self.stop()
        self.tcp_link.close()

class TestMarker(RingBuffer):
    def __init__(self, interval: list, srate: float, events_id): 
        self.interval = interval  
        self.sample_rate = srate
        self.events_id = events_id
        max_size = int(self.interval[1]*srate - self.interval[0]*srate)
        super().__init__(size=max_size)

    def __call__(self, event: int) -> bool:
        m_event = int(event)
        if m_event != 0 and m_event in self.events_id:
            self.cur_event = m_event
            print(event)
            return True
        else:
            return False

    def get_epoch(self):
        index = self.events_id.index(self.cur_event)
        data = super().get_all()

        # data = super().get_all()
        # return data[self.epoch_ind[0]: self.epoch_ind[1]]


        # return data[-2000,:]

class FeedbackWorker(ProcessWorker):
    def __init__(self, run_files, pick_chs, stim_interval, event_map,
                 srate, lsl_source_id, timeout, worker_name):
        self.run_files = run_files
        self.pick_chs = pick_chs
        self.stim_interval = stim_interval
        self.stim_labels = event_map
        self.srate = srate
        self.lsl_source_id = lsl_source_id
        self.data_matlist = []
        self.mode = None
        super().__init__(timeout=timeout, name=worker_name)

    def pre(self):
        X, y, n_block3 = read_data(run_files=self.run_files,
                                 chs=self.pick_chs,
                                 interval=self.stim_interval,
                                 event_map=self.stim_labels,
                                 fs=self.srate)
        # X = X[:,0:6,:]
        # print(X.shape)
        print("Loding train data successfully")
        print("ok is right")
        # Compute offline acc
        self.estimator = offline_validation(X, y, n_block3, srate=self.srate)

        info = StreamInfo(
            name='meta_online_worker',
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
        udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        udp_addr = ('192.168.50.91', 8080)
        
        data = np.array(data, dtype=np.float64).T
        data = data[:-1,:]

        self.data_matlist.append(data)
        np.concatenate(self.data_matlist, axis=0)
        data_matlist = np.array(self.data_matlist)
        print(data_matlist.shape)
        print("data_matlist:", data_matlist[:, 0, :])
        if data_matlist.shape[0] == 16:
            print("save is ok")
            np.save('./Data/Tju-testOffline/1201_11', data_matlist)

        srate = 250
        f0 = 50
        Q = 30
        b_notch, a_notch = signal.iirnotch(f0, Q, fs=srate)

        data = signal.filtfilt(b_notch, a_notch, data)
        p_labels = self.estimator.predict(data)
        p_labels = np.array([int(p_labels + 1)])
        p_labels = p_labels.tolist()
        p_labels = list(p_labels)

        p_labels = list(p_labels)
        print('predict_id_paradigm:{}', p_labels)

        # # udp发送到刺激端   发送p_labels_cca
        # udp_socket.sendto(p_labels[0] - 1, udp_addr)

        # self.outlet。.push_sample(p_labels)
        # while True:
        #     if self.outlet.have_consumers():
        # #   while True:
        # #     self.outlet.push_sample(p_labels)
        #         time.sleep(1)
        #         self.outlet.push_sample(p_labels)
        #         print('self.outlet.have_consumers')
        #         break
        # if self.outlet.have_consumers():
        #     print("send is ok")
        #     self.outlet.push_sample(p_labels)
        # # udp_socket.sendto(p_labels[0],udp_addr)

        k = 0
        my_list = [[1], [2], [3], [4], [5], [6], [7], [8]]
        cre_list.append(p_labels)
        for i in range(len(cre_list)):
            if cre_list[i] == my_list[i % 8]:
                k = k + 1
        print("在线识别准确率：", float(k)/float(len(cre_list)))


    def post(self):
        pass


if __name__ == '__main__':
    # Sample rate EEG amplifier
    srate = 250
    # Data epoch duration, 0.14s visual delay was taken account
    stim_interval = [0.14, 4.64]
    # Label types
    stim_labels = list(range(1, 9))
    event_map = {str(e):e for e in range(1, 9)}
    # stim_labels = 8
    cnts = 1
    # Data path
    run_files = ['C:\\Users\\11402\\Desktop\\EEG_Data\\offline_1208.bdf']
    
    # run_files = ['./Data/Tju-testOffline/1206/offline_text1206.bdf']
    # print(run_files)
    pick_chs = ['P1', 'P2', 'PO3', 'PO4', 'POZ', 'O1', 'O2', 'OZ']

    lsl_source_id = 'mobile-bci'
    feedback_worker_name = 'nano_worker'

    worker = FeedbackWorker(
        run_files=run_files,
        pick_chs=pick_chs,
        stim_interval=stim_interval,
        event_map=event_map, srate=srate,
        lsl_source_id=lsl_source_id,
        timeout=5e-2,
        worker_name=feedback_worker_name)
    marker = Marker(stim_interval,srate,stim_labels)

    # worker.pre()
    # worker.consume(marker.get_epoch())
    # # Set Neuroscan parameters
    
    ns = NanoEEG()
    ns.register_worker(feedback_worker_name, worker, marker)
    ns.up_worker(feedback_worker_name)
    time.sleep(0.5)
    ns.start_trans()

    # # Start tcp connection with ns
    # ns.connect_tcp()
    # # Start acquire data from ns
    # # ns.start_acq()

    # # Register worker for online data processing
    # ns.register_worker(feedback_worker_name, worker, marker)
    # # Start online data processing
    # ns.up_worker(feedback_worker_name)
    # time.sleep(0.5)

    # # Start slicing data and passing data to worker
    # ns.start_trans()

    # input('press any key to close\n')
    # ns.down_worker('feedback_worker')
    # time.sleep(1)

    # # Stop online data retriving of ns
    # ns.stop_trans()
    # ns.stop_acq()
    # ns.close_connection()
    # ns.clear()
    # print('bye')
