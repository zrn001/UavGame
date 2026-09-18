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
from mne.filter import resample
from pylsl import StreamInfo, StreamOutlet
from scipy import signal
from metabci.brainflow.amplifiers import Marker,BaseAmplifier,RingBuffer
from metabci.brainflow.workers import ProcessWorker
from metabci.brainda.algorithms.decomposition.base import (
    generate_filterbank, generate_cca_references)
from metabci.brainda.algorithms.decomposition import FBSCCA, FBTRCA
from sklearn.base import BaseEstimator, ClassifierMixin
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

def cca_train_model():
    
    wp = [[6, 88], [14, 88], [22, 88], [30, 88], [38, 88]
    ]
    ws = [[4, 90], [12, 90], [20, 90], [28, 90], [36, 90]
    ]
    filterweights = np.arange(1, 6)**(-1.25) + 0.25
    filterbank = generate_filterbank(wp, ws, 1000)

    model = FBSCCA(filterbank=filterbank, n_components=5, filterweights=filterweights, n_jobs=-1)

    return model 

class DitingBrainEEGAmplifier(BaseAmplifier):
    LSB = (4.5 * 1_000_000.0) / (1 << 23)

    def __init__(self,
                 device_address: Tuple[str, int] = ('127.0.0.1', 1895),
                 srate=1000,
                 num_chans=64):   # ★ 64导联帽子：64个EEG通道；触发在第65通道(最后一行)
        super().__init__()
        self.device_address = device_address
        self.srate = srate
        self.onePacketSize = 10
        self.num_chans = num_chans
        self.tcp_link = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        self.buffer = b""

        self.payload_format = f"{self.num_chans * self.onePacketSize}i{self.onePacketSize}i"
        self.packet_format = f">i{self.payload_format}"

        self.packet_size = struct.calcsize(self.packet_format)

    def connect_tcp(self):
        self.tcp_link.connect(self.device_address)

    def recv(self):
        while len(self.buffer) < self.packet_size:
            try:
                data = self.tcp_link.recv(8192)
                if not data:
                    print("TCP 连接已断开")
                    return []
                self.buffer += data
            except Exception as e:
                print("接收数据出错:", e)
                self.stop_trans()
                return []

        packet_data = self.buffer[:self.packet_size]
        self.buffer = self.buffer[self.packet_size:]

        # ---- 临时诊断：确认包长/通道数是否匹配（每200包打印一次心跳）----
        self._dbg_count = getattr(self, '_dbg_count', 0) + 1
        if self._dbg_count % 200 == 1:
            print(f"[心跳] 收到第{self._dbg_count}包  期望包长={self.packet_size}字节  "
                  f"num_chans+1={self.num_chans + 1}")
        try:
            return self.__upack_data(packet_data)
        except Exception as e:
            print(f"!!! 解包失败(通道数可能不匹配): {e}  期望包长={self.packet_size}")
            return []

    def __upack_data(self, packet_data):
        unpacked_data = struct.unpack(self.packet_format, packet_data)
        gain = unpacked_data[0]

        payload = unpacked_data[1:]

        data = np.array(payload, dtype=np.float64)
        data = data.reshape((self.num_chans + 1, self.onePacketSize))

        data[:self.num_chans, :] = (data[:self.num_chans, :] * self.LSB) / gain

        # ---- 临时调试：监视触发通道（最后一行）是否有非零标签进来 ----
        trig = data[self.num_chans, :]
        nz = trig[trig != 0]
        if nz.size > 0:
            print(">>> 触发通道非零值:", np.unique(nz))
        # ---- 调试结束 ----

        data = np.transpose(data)

        return data.tolist()

    def start_trans(self):
        self.connect_tcp()
        print("连接成功，开始传输")
        self.start()

    def stop_trans(self):
        self.stop()
        try:
            self.tcp_link.close()
        except:
            pass


class FeedbackWorker(ProcessWorker):
    def __init__(self, run_files, pick_chs, stim_interval, event_map,
                 srate, lsl_source_id, timeout, worker_name, ch_ind):
        self.run_files = run_files
        self.pick_chs = pick_chs
        self.stim_interval = stim_interval
        self.stim_labels = event_map
        self.srate = srate
        self.lsl_source_id = lsl_source_id
        self.data_matlist = []
        self.mode = None
        self.ch_ind = ch_ind
        super().__init__(timeout=timeout, name=worker_name)

    def segment_epoch(self, data, event, tmin=0.0, tmax=1.0, fs=1000):
        """ Segment epochs. """
        # Segment epochs
        n_event, n_chan, n_tpoint = event.shape[0], data.shape[0], int(fs*tmax-fs*tmin)
        epochs = np.zeros((n_event, n_chan, n_tpoint))
        for idx_event in range(n_event):
            idx_t0 = event[idx_event, 0] + int(fs*tmin)
            idx_t1 = event[idx_event, 0] + int(fs*tmax)
            epochs[idx_event] = data[:, idx_t0:idx_t1]
        return epochs

    def read_data(self,run_files, chs, interval, event_map,fs):
        epoch_raw = []
        labels = []
        for idx_bdf in range(len(run_files)):
            # Load data
            # npy_data = np.load('D:\\UserCode\\zhz\\25年杭州终验\\online_new.npy')
            # npy_data = npy_data[0:-1,:]
            raw = mne.io.read_raw_bdf(run_files[idx_bdf], preload=True)
            # Interpolation for bad points
            data = raw.get_data()      
            print(data.shape)
            data = data[:8,:]
            print("data_shape:", data.shape)
            # Event
            event, _ = mne.events_from_annotations(raw, event_id=event_map)

            labels.append(event[:, -1]-1)
            
            # Raw for classification
            data_raw = data.copy()
            epoch_raw.append(self.segment_epoch(data_raw.copy(), event, interval[0], interval[1], fs))
        epoch_raw = np.concatenate(epoch_raw, 0)
        # np.save('./Data/Tju-testOffline/model_ssvep', epoch_raw)
        labels = np.concatenate(labels, 0)
        # print("labels:",labels)
        n_block3 = int(len(labels) / 8)
        # print("n_block3:",n_block3)

        return epoch_raw, labels, n_block3

    def pre(self):
        self.model = cca_train_model() 

        # server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        # # 设置端口重用（可选，避免地址被占用）
        # server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        # # 绑定地址和端口
        # host = '127.0.0.1'  # 监听所有网络接口
        # port = 1890
        # server_socket.bind((host, port))

        # # 开始监听，最大等待连接数5
        # server_socket.listen(5)
        # print(f"服务器启动，监听 {host}:{port}")

        # 接受客户端连接
        # print("等待客户端连接...")
        # self.client_socket, client_address = server_socket.accept()
        # print(f"收到连接: {client_address}")

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
        data = np.array(data, dtype=np.float64).T
        print("原始 data.shape:", data.shape)
        data = data[self.ch_ind]
        f0 = 50
        Q = 30
        b_notch, a_notch = signal.iirnotch(f0, Q, fs=1000)
        data = signal.filtfilt(b_notch, a_notch, data)

        # FBSCCA.predict 期望 3 维 (n_trials, n_channels, n_samples)，
        # 这里补一个 trial 维度：(12, N) → (1, 12, N)
        data = data[np.newaxis, ...]
        print("喂给模型 data.shape:", data.shape)

        freq_list = [8, 9, 10, 11, 12, 13]

        Yf = generate_cca_references(freq_list, srate=1000, T=2, n_harmonics = 5)
        self.model.fit(data,Yf=Yf)
        p_labels_cca = self.model.predict(data)
        p_labels_cca = p_labels_cca+1

        print("p_labels:", p_labels_cca)

        k = 0
        my_list = [[1], [2], [3], [4], [5], [6]]
        cre_list.append(p_labels_cca+1)
        for i in range(len(cre_list)):
            if cre_list[i] == my_list[i % 4]:
                k = k + 1
        acc = float(k) / float(len(cre_list))
        print("在线识别准确率：", acc)

        data = struct.pack('!i', p_labels_cca[0])
        
        # while True:
        #     #发送数据
        #     self.client_socket.send(data)
        #     print(f"已发送响应: {p_labels_cca}")

        #     # self.client_socket.close()
        #     break

        # LSL 
        if self.outlet.have_consumers():
            self.outlet.push_sample(p_labels_cca)

    def post(self):
        try:
            self.udp_sock.close()
        except Exception:
            pass


if __name__ == '__main__':

    # Sample rate EEG amplifier
    srate = 1000
    # Data epoch duration, 0.14s visual delay was taken account
    stim_interval = [0.14, 2.14]
    # Label types
    stim_labels = list(range(1, 7))
    event_map = {str(e):e for e in range(1, 255)}
    # stim_labels = 8
    cnts = 1
    # Data path
    run_files = ['./Data/Tju-testOffline/1130/S2.bdf']
    run_files = ['C:\\Users\\DELL\\Desktop\\移动\\0408\\offline2.bdf']
    pick_chs = ['PZ', 'PO5', 'PO3', 'POZ', 'PO4', 'PO6', 'O1', 'OZ', 'O2']
    # ★ 64导联帽子的枕-顶区 SSVEP 导联（1-based 物理序号，见下方通道表）：
    #   48 PZ  54 PO5  55 PO3  56 POZ  57 PO4  58 PO6  61 O1  62 OZ  63 O2
    # 64导联顺序：FP1 FPZ FP2 AF3 AF4 F7 F5 F3 F1 FZ F2 F4 F6 F8 FT7 FC5 FC3
    #   FC1 FCZ FC2 FC4 FC6 FT8 T7 C5 C3 C1 CZ C2 C4 C6 T8 M1 TP7 CP5 CP3
    #   CP1 CPZ CP2 CP4 CP6 TP8 M2 P7 P5 P3 P1 PZ P2 P4 P6 P8 PO7 PO5 PO3
    #   POZ PO4 PO6 PO8 AF7 O1 OZ O2 AF8 
    ch_ind = np.array([48, 54, 55, 56, 57, 58, 61, 62, 63], dtype=int) - 1  # 真实选择导联

    lsl_source_id = 'meta_online_worker666'
    feedback_worker_name = 'feedback_worker'
    udp_target = ('172.21.20.107', 7810)

    worker = FeedbackWorker(
        run_files=run_files,
        pick_chs=pick_chs,
        stim_interval=stim_interval,
        event_map=event_map, srate=srate,
        lsl_source_id=lsl_source_id,
        timeout=5e-2,
        worker_name=feedback_worker_name, ch_ind = ch_ind)
    marker = Marker(stim_interval,srate,stim_labels)

    # worker.pre()
    # worker.consume(marker.get_epoch())
    # # Set Neuroscan parameters
    
    ns = DitingBrainEEGAmplifier()
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
