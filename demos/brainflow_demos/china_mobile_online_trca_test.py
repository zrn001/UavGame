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
import datetime
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
from metabci.brainda.algorithms.decomposition import FBTDCA, FBTRCA, FBSCCA, SCCA
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


def leave_one_block_out_test(data, label, n_block):
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
        f0 = 50
        Q = 30
        b_notch, a_notch = signal.iirnotch(f0, Q, fs=1000)
        data_train = signal.filtfilt(b_notch, a_notch, data_train)
        
        model = train_model(data_train,label_train)
        label_pre = model.predict(data_test) 
        result.append(np.mean(label_pre == label_test))
        # Get scores
        # result.append(model.score(data_test, label_test))
    result = np.stack(result, axis=0)
    return result


'''-----------------------In[5.1]数据滤波处理--------------------------'''
def bandpass(sig, freq0, freq1, srate, axis=-1):
    wn1 = 2*freq0/srate
    wn2 = 2*freq1/srate
    b, a = signal.butter(4, [wn1, wn2], 'bandpass')
    srate = 1000
    f0 = 50
    Q = 30
    b_notch, a_notch = signal.iirnotch(f0, Q, fs=srate)
    sig_new = signal.filtfilt(b_notch, a_notch, sig)
    sig_new = signal.filtfilt(b, a, sig, axis=axis)
    return sig_new


'''-----------------------In[5.2]数据重采样处理--------------------------'''
def resample(X, srate_up, srate_down):
    X = np.reshape(X, (-1, X.shape[-2], X.shape[-1]))
    X = resample(X, up=srate_up, down=srate_down)


'''-----------------------In[5.3]数据片段划分--------------------------'''
def segment_epoch(data, event, tmin=0.0, tmax=1.0, fs=1000):
    """ Segment epochs. """
    # Segment epochs
    n_event, n_chan, n_tpoint = event.shape[0], data.shape[0], int(fs*tmax-fs*tmin)
    epochs = np.zeros((n_event, n_chan, n_tpoint))
    for idx_event in range(n_event):
        idx_t0 = event[idx_event, 0] + int(fs*tmin)
        idx_t1 = event[idx_event, 0] + int(fs*tmax)
        epochs[idx_event] = data[:, idx_t0:idx_t1]
    return epochs

'''-----------------------In[5.4]数据基线矫正--------------------------'''
def Baseline_Corr(X):
    X = np.reshape(X, (-1, X.shape[-2], X.shape[-1]))
    X = X - np.mean(X, axis=-1, keepdims=True)
    X = X / np.std(X, axis=(-1, -2), keepdims=True)


'''-----------------------In[6.1]TRCA算法进行特征提取与建模,模型数据格式为.npy--------------------------'''
def read_data(run_files, chs, interval, event_map,fs):
    epoch_raw = []
    labels = []
    for idx_bdf in range(len(run_files)):
        # Load data
        # npy_data = np.load('D:\\UserCode\\zhz\\25年杭州终验\\online_new.npy')
        # npy_data = npy_data[0:-1,:]
        raw = mne.io.read_raw_bdf(run_files[idx_bdf], preload=True)
        # raw.pick_channels(chs)
        # Interpolation for bad points
        data = raw.get_data()      
        print(data.shape)
        # data = data[:8,:]
        print("data_shape:", data.shape)
        # Event
        event, _ = mne.events_from_annotations(raw, event_id=event_map)

        labels.append(event[:, -1]-1)
        
        # Raw for classification
        data_raw = data.copy()
        epoch_raw.append(segment_epoch(data_raw.copy(), event, interval[0], interval[1], fs))
    epoch_raw = np.concatenate(epoch_raw, 0)
    # np.save('./Data/Tju-testOffline/model_ssvep', epoch_raw)
    labels = np.concatenate(labels, 0)
    # print("labels:",labels)
    n_block3 = int(len(labels) / 5)
    # print("n_block3:",n_block3)

    return epoch_raw, labels, n_block3


'''-----------------------In[7.1]可实现CCA、FBCCA、TRCA算法进行模式识别------------------'''
def train_model(X, Y):
    srate = 1000
    # TRCA decoding model
    # wp = [(8,48), (16,48), (24,48)]
    # ws = [(6,50), (14,50), (22,50)]
    wp = [(5,90), (14,90), (23,90), (32,90), (41,90)]
    ws = [(3,92), (12,92), (21,92), (30,92), (39,92)]
    # wp = [[6, 88], [14, 88], [22, 88], [30, 88], [38, 88]
    # ]
    # ws = [[4, 90], [12, 90], [20, 90], [28, 90], [36, 90]
    # ]

    filterweights = np.arange(1, 6)**(-1.25) + 0.25
    filterbank = generate_filterbank(wp, ws, srate)
    # filterbank = generate_filterbank(wp, ws, order=15, srate=srate, rp=0.5)

    model = FBTRCA(filterbank=filterbank, n_components=5, ensemble=True, filterweights=filterweights, n_jobs=-1)
    # model = FBSCCA(filterbank=filterbank, n_components=5, filterweights=filterweights, n_jobs=-1)
    # mode = SCCA(n_components=5, n_jobs=-1)
    # Fit model
    model.fit(X, Y)
    return model


def model_predict(X, srate=1000, model=None):
    srate = 1000
    f0 = 50
    Q = 30
    b_notch, a_notch = signal.iirnotch(f0, Q, fs=srate)
    X = signal.filtfilt(b_notch, a_notch, X)

    X = np.reshape(X, (-1, X.shape[-2], X.shape[-1]))
    # print("X-shape:",X.shape)
    # print("X:",X)

    # print(X)
    print("X.shape: ",X.shape)
    p_labels = model.predict(X)    # EEG data, shape(n_trials, n_channels, n_samples).
    return p_labels


def offline_validation(X, y, n_block3, srate=1000):
    t_list = np.round(np.arange(0.5, 1.05, 0.1), 1)
    acc = np.zeros((len(t_list)))
    # model, b_notch, a_notch  = train_model()
    for idx_t, t in enumerate(t_list):
        t0 = time.time()
        # Notch filtering for suit time length
        # X = signal.filtfilt(b_notch, a_notch, X)

        # Leave-one-block-out cross validation
        result = leave_one_block_out_test(X[..., :int(t*srate)], y, n_block=n_block3)
        # Accuracy
        acc[idx_t] = np.mean(result)
        print('Time:{} Acc:{:.3f}'.format(t, acc[idx_t]))
        
        b_notch, a_notch = signal.iirnotch(50, 30, fs=srate)
        epoch = signal.filtfilt(b_notch, a_notch, X[..., :int(t*srate)])
        model_finally = train_model(epoch,y)
        print('Training time:{:.4f}'.format(time.time()-t0))
        
    return model_finally


class NanoEEG(BaseAmplifier):

    def __init__(self,
                 device_address: Tuple[str, int] = ('127.0.0.1', 1895),
                 srate=1000,
                 num_chans=32):
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
            # print("okkkkkkkk")
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
                # print("ok")
                # print(data)
                # self.all_data.append(data)
                # print("all_data:", self.all_data)
                # print("all_data:", self.all_data)
                # if len(data) !=10 or len(data[0])!=9:
                #     print('Len: {}, {}'.format(len(data), len(data[0])))
                # if self.data_count == 250:
                #     joblib.dump(self.all_data, "all_data.pkl")
                return data
            else:
                print("连接出错,此次请求数据失败，返回空数据")
                return []

    def __upack_data(self, data):
        # 一次十个采样点的数据 每个为int24 3个字节 (导联数+1)*10*3
        data = struct.unpack(f"{(self.num_chans + 1) * self.srate*40//1000}i", data)
        data = np.array(data, dtype=np.int32)
        data = data.reshape((self.num_chans + 1, self.srate*40//1000))
        data = np.transpose(data)
        # print("ok:", self.srate*40//1000)   # 每次接受40个包了
        # data_label = data[:, -1]
        # self.all_data.append(data_label)
        # if len(self.all_data) == 1000:
        #     print("save is ok")
        #     np.save('D:\\UserCode\\zhz\\25年杭州终验\\online_new.npy', self.all_data)
        # print("label:", self.all_data)

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
        print(X.shape)
        print("Loding train data successfully")
    
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
        # start time 
        print("consume")

        raw_data = data
        f0 = 50
        Q = 30
        b_notch, a_notch = signal.iirnotch(f0, Q, fs=self.srate)
        data = signal.filtfilt(b_notch, a_notch, raw_data)
        data = np.array(data, dtype=np.float64).T
        print(data.shape)
        data = data[:32,:]   # 16个导联
        
        p_labels_trca = model_predict(data, srate=self.srate, model=self.estimator)
        p_labels_trca += 1
        print("p_labels:", p_labels_trca)

        k = 0
        my_list = [[0], [1], [2], [3], [4], [5], [6], [7]]
        cre_list.append(p_labels_trca)
        for i in range(len(cre_list)):
            if cre_list[i] == my_list[i % 8]:
                k = k + 1
        print("在线识别准确率：", float(k)/float(len(cre_list)))

    def post(self):
        pass


if __name__ == '__main__':
    # Sample rate EEG amplifier
    srate = 1000
    # Data epoch duration, 0.14s visual delay was taken account
    stim_interval = [0.14, 1.14]
    # Label types
    stim_labels = list(range(1, 6))
    event_map = {str(e):e for e in range(1, 255)}
    # stim_labels = 8
    cnts = 1
    # Data path5s-trca_offline2
    # run_files = ['./Data/Tju-testOffline/1130/S2.bdf']
    run_files = [r"E:\脑控无人机\5-BrainDrone-TJU\test1.bdf"]
    # run_files = ['./Data/Tju-testOffline/1130/S2.bdf']
    print(run_files)
    # pick_chs = ['P1', 'P2', 'PO3', 'PO4', 'POZ', 'O1', 'O2', 'OZ']
    pick_chs = ['F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T7', 'T8', 'P7', 'P8', 'POz']

    lsl_source_id = 'mobile-bci'
    feedback_worker_name = 'feedback_worker'

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
