# -*- coding: utf-8 -*-
# License: MIT License
"""
SSAVEP Feedback on NeuroScan.

"""
import time
import numpy as np
import struct
from typing import Tuple
import socket
import seaborn as sns

import mne
from mne.filter import resample
import datetime
from pylsl import StreamInfo, StreamOutlet
from metabci.brainflow.amplifiers import NeuroScan, BaseAmplifier, Marker,RingBuffer
from metabci.brainflow.workers import ProcessWorker
from metabci.brainda.algorithms.decomposition.base import generate_filterbank
from metabci.brainda.algorithms.utils.model_selection \
    import EnhancedLeaveOneGroupOut
from metabci.brainda.algorithms.decomposition.csp import FBCSP, CSP
from metabci.brainda.utils import upper_ch_names
from mne.io import read_raw_cnt
from sklearn.svm import SVC
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import make_pipeline
from scipy import signal
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
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

'''-----------------------In[5.1]数据滤波处理--------------------------'''
def bandpass(sig, freq0, freq1, srate, axis=-1):
    wn1 = 2*freq0/srate
    wn2 = 2*freq1/srate
    b, a = signal.butter(4, [wn1, wn2], 'bandpass')
    srate = 250
    f0 = 50
    Q = 30
    b_notch, a_notch = signal.iirnotch(f0, Q, fs=srate)
    # sig_new = signal.filtfilt(b_notch, a_notch, sig_new)
    sig_new = signal.filtfilt(b, a, sig, axis=axis)
    return sig_new


'''-----------------------In[5.2]数据重采样处理--------------------------'''
def resample(X, srate_up, srate_down):
    X = np.reshape(X, (-1, X.shape[-2], X.shape[-1]))
    X = resample(X, up=srate_up, down=srate_down)


'''-----------------------In[5.3]数据片段划分--------------------------'''
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

'''-----------------------In[5.4]数据基线矫正--------------------------'''
def Baseline_Corr(X):
    X = np.reshape(X, (-1, X.shape[-2], X.shape[-1]))
    X = X - np.mean(X, axis=-1, keepdims=True)
    X = X / np.std(X, axis=(-1, -2), keepdims=True)


'''-----------------------In[6.2]采用FBCSP算法进行特征提取与建模,模型数据格式为.npy--------------------------'''
def read_data(run_files, chs, interval, labels, delay):
    # print(run_files)
    Xs, ys = [], []
    for run_file in run_files:
        raw = mne.io.read_raw_bdf(run_file, preload=True, verbose=False)
        raw = raw.filter(l_freq=6, h_freq=48, method='iir')
        raw = raw.notch_filter(50)
        # chs = ['19', '16', '08', '02', '01', '11', '06', '18']
        # chs = ['CH1', 'CH2', 'CH3', 'CH4', 'CH5', 'CH6', 'CH7', 'CH8']
        ch_picks = mne.pick_channels(raw.ch_names, chs, ordered=True)
        events,b = mne.events_from_annotations(raw)
        events[:,2] -= 1
        epochs = mne.Epochs(raw, events, event_id=labels, tmin=interval[0] + delay, tmax=interval[1] + delay,
                            baseline=None, picks=ch_picks, verbose=False)
        np.save('./Data/Tju-testOffline/model_mi', epochs)

        for label in labels:
            X = epochs[str(label)].get_data()[..., 1:]  # 2000 points
            Xs.append(X)
            ys.append(np.ones((len(X))) * label)
    Xs = np.concatenate(Xs, axis=0)
    ys = np.concatenate(ys, axis=0)
    ys = label_encoder(ys, labels)


    return Xs, ys, ch_picks

'''-----------------------In[7.2]可实现CSP、FBCSP算法进行模式识别--------------------------'''
def train_model(X, y, srate=1000):
    y = np.reshape(y, (-1))

    wp = [(4, 8), (8, 12), (12, 30)]
    ws = [(2, 10), (6, 14), (10, 32)]
    filterbank = generate_filterbank(wp, ws, srate=250, order=4, rp=0.5)
    # model = make_pipeline(
    #     MultiCSP(n_components = 2),
    #     LinearDiscriminantAnalysis())
    model = make_pipeline(*[
        FBCSP(n_components=5,
              n_mutualinfo_components=4,
              filterbank=filterbank),
        SVC()
    ])
    
    # model = make_pipeline(*[
    #     CSP(n_components=5,
    #           n_mutualinfo_components=4),
    #     SVC()
    # ])
    # fit()训练模型
    model = model.fit(X, y)

    return model

# 预测标签


def model_predict(X, srate=250, model=None):
    X = np.reshape(X, (-1, X.shape[-2], X.shape[-1]))
    # 降采样
    # X = resample(X, up=256, down=srate)
    # 滤波
    X = bandpass(X, 6, 48, 250)
    # 零均值单位方差 归一化
    # X = X - np.mean(X, axis=-1, keepdims=True)
    # X = X / np.std(X, axis=(-1, -2), keepdims=True)
    # predict()预测标签
    p_labels = model.predict(X)
    return p_labels

# 计算离线正确率


def offline_validation(X, y, srate=250):
    y = np.reshape(y, (-1))
    # print(y)
    spliter = EnhancedLeaveOneGroupOut(return_validate=False)

    kfold_accs = []
    ytest=np.zeros((y.size,1))
    plabel=np.zeros((y.size,1))
    for train_ind, test_ind in spliter.split(X, y=y):
        X_train, y_train = np.copy(X[train_ind]), np.copy(y[train_ind])
        X_test, y_test = np.copy(X[test_ind]), np.copy(y[test_ind])

        model = train_model(X_train, y_train, srate=srate)
        p_labels = model_predict(X_test, srate=srate, model=model)
        kfold_accs.append(np.mean(p_labels == y_test))
        ytest[test_ind,:] = np.reshape(y_test,(len(y_test),1))
        plabel[test_ind,:] = np.reshape(p_labels,(len(p_labels),1))
        
    cm = confusion_matrix(ytest, plabel)
    plt.figure()
    sns.heatmap(cm,annot=True)
    plt.show()    
    # print(kfold_accs)
    return np.mean(kfold_accs)


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

    def connect_tcp(self):
        self.tcp_link.connect(self.device_address)

    def recv(self):
        data = None
        try:
            data = self.tcp_link.recv(9216)
        except Exception as e:
            print("连接出错", e)
            self.tcp_link.close()
        finally:
            # 
            if data is not None:
                # print('finally1')
                data = self.__upack_data(data)
                # print(data)
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
        # print("缓冲区大小",np.array(data).shape)
        # 返回往前的4个点的数据
        return data[-1000:]

class FeedbackWorker(ProcessWorker):
    def __init__(self,
                 run_files,
                 pick_chs,
                 stim_interval,
                 stim_labels,
                 srate,
                 lsl_source_id,
                 timeout,
                 worker_name):
        self.run_files = run_files
        self.pick_chs = pick_chs
        self.stim_interval = stim_interval
        self.stim_labels = stim_labels
        self.srate = srate
        self.lsl_source_id = lsl_source_id
        super().__init__(timeout=timeout, name=worker_name)

    def pre(self):

        # srate = self.srate
        # labels = self.stim_labels
        interval = self.stim_interval
        # down_srate = 250

        X, y, ch_ind = read_data(run_files=self.run_files,
                                 chs=self.pick_chs,
                                 interval=self.stim_interval,
                                 labels=self.stim_labels,delay=0)
        # print(X.shape)
        # X = X[:,0:8,:]
        print(X.shape)
        # X = resample(X, up=down_srate, down=srate)
        print("Loding data successfully")
        acc = offline_validation(X, y, srate=self.srate)     # 计算离线准确率
        print("Current Model accuracy:", acc)
        self.estimator = train_model(X, y, srate=self.srate)
        self.ch_ind = ch_ind
        # print(ch_ind)
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
        start_decodeEEG = datetime.datetime.now()

        udp_socket2Robot = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        udp_addr2Robot = ('192.168.50.19', 8080)

        udp_socket2Stim = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        udp_addr2Stim = ('192.168.50.91', 8001)

        data = np.array(data, dtype=np.float64).T
        # print(data.shape)
        data = data[0:8,:]
        # print(data.shape)

        p_labels = model_predict(data, srate=self.srate, model=self.estimator)
        p_labels = int(p_labels)
        p_labels = p_labels + 1

        end_decodeEEG = datetime.datetime.now()
        during = ((end_decodeEEG -start_decodeEEG).seconds * 1000 + (end_decodeEEG -start_decodeEEG).microseconds / 1000)
        print("EEG_decodeTime: ", during, "ms (0~500ms)")

        # end time 

        p_labels = [p_labels]
        label_send = struct.pack('i', p_labels[0])
        # udp发送到刺激端   发送p_labels_cca
        udp_socket2Robot.sendto(label_send, udp_addr2Robot)
        
        time.sleep(0.5)
        for i in range(100):
            udp_socket2Stim.sendto(label_send, udp_addr2Stim)
        # p_labels = p_labels.tolist()
        print(p_labels)
        

        if self.outlet.have_consumers():
            self.outlet.push_sample(p_labels)
        k = 0
        my_list = [[1], [2]]
        cre_list.append(p_labels)
        for i in range(len(cre_list)):
            if cre_list[i] == my_list[i % 2]:
                k = k + 1
        print("在线识别准确率：", float(k)/float(len(cre_list)))

    def post(self):
        pass


if __name__ == '__main__':
    srate = 250
    # labels = list(range(1, 5))
    run_files = ['C:\\Users\\11402\\Desktop\\EEG_Data\\mi_text1208.bdf']
    #'C:\\Users\\DELL\\Desktop\\1129\\5.bdf'
    #'C:\\Users\\DELL\\Desktop\\1129\\6.bdf'
    # 预测所用时间
    stim_interval = [0, 3.5]
    # Label types
    stim_labels = list(range(1,3))  # stim_labels = list(range(1, 5))
    # print(stim_labels)
    # cnts = 3
    # Data path
    # filepath = "data\\train\\sub1"
    # runs = list(range(1, cnts+1))
    # run_files = ['{:s}\\{:d}.cnt'.format(
    #     filepath, run) for run in runs]

    # marker = Marker(interval=stim_interval, srate=srate,
    #                 events=stim_labels)
    pick_chs = ['P1', 'P2', 'PO3', 'PO4', 'POz', 'O1', 'O2', 'Oz']

    lsl_source_id = 'mobile-bci'
    feedback_worker_name = 'nano_worker'
    worker = FeedbackWorker(
        run_files=run_files,
        pick_chs=pick_chs,
        stim_interval=stim_interval,
        stim_labels=stim_labels, srate=srate,
        lsl_source_id=lsl_source_id,
        timeout=1e-3,
        worker_name=feedback_worker_name)

    marker = Marker(stim_interval,srate,stim_labels)

    ns = NanoEEG()
    ns.register_worker(feedback_worker_name, worker, marker)
    ns.up_worker(feedback_worker_name)
    time.sleep(0.5)
    ns.start_trans()
    # ns = MicroCollect(
    #     device_address=('127.0.0.1', 2777),
    #     srate=srate,
    #     num_chans=8)

    # # 与ns建立tcp连接
    # ns.connect_tcp()
    # # ns开始采集波形数据
    # # ns.start_acq() 

    # # register worker来实现在线处理
    # ns.register_worker(feedback_worker_name, worker, marker)
    # # 开启在线处理进程
    # ns.up_worker(feedback_worker_name)
    # # 等待 0.5s
    # time.sleep(0.5)

    # # ns开始截取数据线程，并把数据传递数据给处理进程
    # ns.start_trans()

    # 任意键关闭处理进程
    # input('press any key to close\n')
    # # 关闭处理进程
    # ns.down_worker('feedback_worker')
    # # 等待 1s
    # time.sleep(1)
    #
    # # ns停止在线截取线程
    # ns.stop_trans()
    # # # ns停止采集波形数据
    # # ns.stop_acq()
    # ns.close_connection()  # 与ns断开连接
    # ns.clear()
    # print('bye')
