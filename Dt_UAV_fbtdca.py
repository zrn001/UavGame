# -*- coding: utf-8 -*-
"""
SSVEP (FBTDCA) 在线反馈 —— 谛听脑电放大器 (DitingBrainEEGAmplifier) 版本。

由 CCA_BrainCar.py 改造而来，两处与 Neuroscan_online_fbtdca_grit.py 对齐：
1. SSVEP 分类算法：FBSCCA（在线实时拟合）-> FBTDCA（filterbank TDCA，离线训练 + 在线预测）；
2. 反馈发送方式：LSL -> UDP socket（默认 127.0.0.1:8888，发送模型预测标签）。

其余（设备连接、Marker/Worker 框架）沿用 CCA_BrainCar.py 的设定；
刺激参数沿用 Neuroscan_online_fbtdca_grit.py 的设定。
"""

import datetime
import os
import socket
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import mne
from mne.filter import resample
from mne.io import read_raw_bdf, read_raw_cnt
from typing import List, Optional, Tuple, Dict, Any
import struct

from metabci.brainflow.amplifiers import BaseAmplifier, Marker
from metabci.brainflow.workers import ProcessWorker
from metabci.brainda.algorithms.decomposition import FBTDCA
from metabci.brainda.algorithms.decomposition.base import (
    generate_filterbank, generate_cca_references)
from metabci.brainda.algorithms.utils.model_selection import (
    EnhancedLeaveOneGroupOut)
from metabci.brainda.utils import upper_ch_names

# ---------------- 刺激参数（沿用 Neuroscan_online_fbtdca_grit.py） ----------------
FREQS = [9.6, 11.6, 9.2,12.8, 10.4, 8.8, 13.2, 10.8, 10.0, 11.2, 12.0, 12.4]
PHASES = [1.4, 1.15, 1.05, 0.0, 0.1, 0.7, 0.35, 0.45, 1.75, 0.8, 1.45, 1.8]

class DitingBrainEEGAmplifier(BaseAmplifier):
    LSB = (4.5 * 1_000_000.0) / (1 << 23)

    def __init__(self,
                 device_address: Tuple[str, int] = ('127.0.0.1', 1895),
                 srate=1000,
                 num_chans=64):
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

        return self.__upack_data(packet_data)

    def __upack_data(self, packet_data):
        unpacked_data = struct.unpack(self.packet_format, packet_data)
        gain = unpacked_data[0]
        payload = unpacked_data[1:]

        data = np.array(payload, dtype=np.float64)
        data = data.reshape((self.num_chans + 1, self.onePacketSize))
        data[:self.num_chans, :] = (data[:self.num_chans, :] * self.LSB) / gain
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



def label_encoder(y, labels):
    new_y = y.copy()
    for i, label in enumerate(labels):
        ix = (y == label)
        new_y[ix] = i
    return new_y


# ---------- SSVEP 数据读取 ----------
def read_data(run_files, chs, interval, labels):
    """读取离线训练数据（支持 .cnt / .bdf），流程与 Neuroscan 版本一致。"""
    Xs, ys = [], []
    for run_file in run_files:
        run_file = Path(run_file)
        raw = mne.io.read_raw_bdf(run_file, preload=True, 
                                  stim_channel="Trigger/Status", verbose=False)
        events = mne.find_events(raw, stim_channel="Trigger/Status", 
                                 shortest_event=1, mask=255, mask_type='and', verbose=False)
        
        ch_picks = mne.pick_channels(raw.ch_names, chs, ordered=True)
        epochs = mne.Epochs(raw, events, event_id=labels,
                            tmin=interval[0], tmax=interval[1],
                            baseline=None, picks=ch_picks, verbose=False)
        for label in labels:
            X = epochs[str(label)].get_data()[..., 1:]
            Xs.append(X)
            ys.append(np.ones((len(X))) * label)
    Xs = np.concatenate(Xs, axis=0)
    ys = np.concatenate(ys, axis=0)
    ys = label_encoder(ys, labels)
    return Xs, ys, ch_picks


# ---------- SSVEP 模型训练 ----------
def train_model(X, y, srate=1000):
    """训练 FBTDCA 模型，流程与 Neuroscan 版本一致。"""
    y = np.reshape(y, (-1))
    X = resample(X, up=256, down=srate)

    # 5 子带 filterbank
    wp = [[6, 88], [14, 88], [22, 88], [30, 88], [38, 88]]
    ws = [[4, 90], [12, 90], [20, 90], [28, 90], [36, 90]]
    filterweights = np.arange(1, 6) ** (-1.25) + 0.25
    filterbank = generate_filterbank(wp, ws, 256)

    # Z-score 归一化
    X = X - np.mean(X, axis=-1, keepdims=True)
    X = X / np.std(X, axis=(-1, -2), keepdims=True)

    # 生成 CCA 参考模板
    freqs = np.array(FREQS)
    Yf = generate_cca_references(freqs, srate=256, T=1, n_harmonics=5)

    model = FBTDCA(filterbank, padding_len=3, n_components=4,
                   filterweights=np.array(filterweights))
    model = model.fit(X, y, Yf)
    return model


# ---------- SSVEP 在线预测 ----------
def model_predict(X, srate=1000, model=None):
    """FBTDCA 在线预测，流程与 Neuroscan 版本一致。"""
    X = np.reshape(X, (-1, X.shape[-2], X.shape[-1]))
    X = X[..., 1:]          # 与 read_data 保持一致，去掉第一个采样点
    X = resample(X, up=256, down=srate)
    X = X - np.mean(X, axis=-1, keepdims=True)
    X = X / np.std(X, axis=(-1, -2), keepdims=True)
    p_labels = model.predict(X)
    return p_labels


def offline_validation(X, y, srate=1000):
    """交叉验证评估 FBTDCA 模型准确率。"""
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
    """SSVEP 在线反馈 worker：FBTDCA 分类 + UDP socket 发送标签。

    Parameters
    ----------
    run_files : list[str]
        离线训练数据文件路径（支持 .cnt / .bdf）。
    pick_chs : list[str]
        参与识别的导联名称。
    stim_interval : list[float]
        单次刺激对应的数据截取时间窗，单位秒。
    stim_labels : list[int]
        刺激事件标签。
    srate : int
        EEG 采样率，单位 Hz。
    timeout : float
        Worker 等待数据的超时时间，单位秒。
    worker_name : str
        Worker 注册到放大器时使用的名称。
    ipFb : str
        UDP 反馈接收端 IP 地址。
    portFb : int
        UDP 反馈接收端端口号。
    save_path : str or pathlib.Path
        在线 epoch 数据保存路径，保存结果的形状为
        (n_epochs, n_channels, n_samples)。
    """

    def __init__(self, run_files, pick_chs, stim_interval, stim_labels,
                 srate, timeout, worker_name,
                 ipFb='127.0.0.1', portFb=8888,
                 save_path='online_epochs.npy'):
        self.run_files = run_files
        self.pick_chs = pick_chs
        self.stim_interval = stim_interval
        self.stim_labels = stim_labels
        self.srate = srate
        self.ipFb = ipFb
        self.portFb = portFb
        self.save_path = Path(save_path)
        self._epochs = []
        super().__init__(timeout=timeout, name=worker_name)

    def pre(self):
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        self._epochs = []
        X, y, ch_ind = read_data(run_files=self.run_files,
                                 chs=self.pick_chs,
                                 interval=self.stim_interval,
                                 labels=self.stim_labels)
        self.ch_ind = ch_ind
        print('ch_ind for feedback worker:', ch_ind)
        print("Loading train data successfully")

        # 交叉验证准确率
        acc = offline_validation(X, y, srate=self.srate)
        print("Current Model accuracy:{:.2f}".format(acc))

        self.estimator = train_model(X, y, srate=self.srate)

        self.socket_fb = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        print('model ok! Connected')

    def _save_epochs(self):
        """将当前已接收的全部 epoch 原子写入单个 npy 文件。"""
        epochs = np.stack(self._epochs, axis=0)
        temp_path = self.save_path.with_name(
            self.save_path.name + '.tmp')
        with open(temp_path, 'wb') as file:
            np.save(file, epochs, allow_pickle=False)
        os.replace(temp_path, self.save_path)

    def consume(self, data):
        data = np.array(data, dtype=np.float64).T
        self._epochs.append(data.copy())
        self._save_epochs()
        print('已保存 epoch {} 到 {}'.format(
            len(self._epochs), self.save_path))
        event_data = data[-1, :]
        event_label = int(np.max(event_data))
        if event_label != 0:
            print("收到事件标签:", event_label)
        data = data[self.ch_ind]
        p_labels = model_predict(data, srate=self.srate, model=self.estimator)
        p_label = int(np.ravel(p_labels)[0])
        print("预测（发送）结果为:", p_label)

        now = datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3]
        msg = bytes(str(p_label), encoding='utf-8')
        self.socket_fb.sendto(msg, (self.ipFb, self.portFb))
        print("p_labels send done at {}".format(now))
        print("********************************************\r\n")

    def post(self):
        try:
            self.socket_fb.close()
        except Exception:
            pass


if __name__ == '__main__':

    srate = 1000
    # 单次刺激的数据截取时间窗，单位秒；0.14 s 用于补偿视觉呈现延迟。
    stim_interval = [0.14, 1.18]
    pick_chs = ['PO5', 'PO3', 'POZ', 'PO4', 'PO6', 
                'O1', 'OZ', 'O2','PO7','PO8',
                'PZ','P1','P2','P3','P4',
                'P5','P6','P7','P8']
    stim_labels = list(range(1, len(FREQS) + 1))
    print("刺激标签:", stim_labels)

    # 离线训练数据（支持 .cnt / .bdf，按实际数据路径修改）
    # 注意：训练数据的通道顺序需与在线放大器输出的通道顺序一致
    run_files = ['./offline_data/ltf_offline1.bdf']

    feedback_worker_name = 'feedback_worker'

    # ---------------- SSVEP Feedback Worker ----------------
    worker = FeedbackWorker(
        run_files=run_files,
        pick_chs=pick_chs,
        stim_interval=stim_interval,
        stim_labels=stim_labels,
        srate=srate,
        timeout=5e-2,
        worker_name=feedback_worker_name,
        save_path=PROJECT_ROOT / 'online_epochs.npy')
    marker = Marker(interval=stim_interval, srate=srate,
                    events=stim_labels)

    # ---------------- 谛听脑电放大器 ----------------
    # 当前帽子为 64 导（num_chans=64），如需其他导联数请修改放大器对应的导联数
    ns = DitingBrainEEGAmplifier()
    ns.register_worker(feedback_worker_name, worker, marker)
    ns.up_worker(feedback_worker_name)
    time.sleep(0.5)
    ns.start_trans()

    input('press any key to close\n')
    ns.down_worker(feedback_worker_name)
    time.sleep(0.5)
    ns.stop_trans()
    ns.clear()
    print('bye')


'''
git config --global --add safe.directory "G:/无人机比赛/DiTing_BrainCar-main"
'''