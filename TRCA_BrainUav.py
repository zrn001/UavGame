# -*- coding: utf-8 -*-
"""
基于 TRCA 的 SSVEP 脑控小车在线反馈系统。
相比 CCA 版本的核心改动：
  1. 算法：CCA → FBTRCA（需离线校准数据训练）
  2. 时间窗口：2.0s → 0.5s（stim_interval [0.14, 0.64]）
  3. 分类频率：每秒 2 次（每 0.5 秒产生一个分类结果）

架构：
  离线阶段：读取 BDF 校准文件 → 切 0.5s epoch → 训练 TRCA 模型
  在线阶段：TCP 接收脑电 → 陷波 → TRCA 预测 → LSL 发送结果
"""

import os
import sys
import time
import pickle
import datetime
import numpy as np
import socket
import struct
from typing import Tuple
from pynput.keyboard import Key, KeyCode, Controller

# 让 print 的中文/emoji 在 Windows GBK 终端也能正常输出，不再因编码崩溃
try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass

import mne
from mne.filter import resample
from pylsl import StreamInfo, StreamOutlet
from scipy import signal
from metabci.brainflow.amplifiers import Marker, BaseAmplifier, RingBuffer
from metabci.brainflow.workers import ProcessWorker
from metabci.brainda.algorithms.decomposition.base import (
    generate_filterbank, generate_cca_references)
from metabci.brainda.algorithms.decomposition import FBTRCA
from sklearn.base import BaseEstimator, ClassifierMixin
from utils.extract_events import extract_events, read_raw_bdf
# ★ 从共享配置读取：坏导联、分类窗长（改配置只需改 config_channels.py）
from config_channels import GOOD_CH_INDEX, STIM_INTERVAL, print_config


# ============================================================================
# UDP 地址配置（反馈发送目标）
# ============================================================================

# 反馈目标：无人机/界面接收分类结果的地址
FEEDBACK_IP = '127.0.0.1'
FEEDBACK_PORT = 7823

# 回环控制目标：发给本进程，主程序收到后模拟键盘按键
CTL_IP = '127.0.0.1'
CTL_PORT = 9999


# ============================================================================
# 工具函数
# ============================================================================

def label_encoder(y, labels):
    """将标签映射为 0~N-1 的连续整数"""
    new_y = y.copy()
    for i, label in enumerate(labels):
        ix = (y == label)
        new_y[ix] = i
    return new_y


class MaxClassifier(BaseEstimator, ClassifierMixin):
    """最大投票分类器（备用）"""
    def __init__(self):
        pass

    def fit(self, X, y):
        pass

    def predict(self, X):
        X = X.reshape((-1, X.shape[-1]))
        y = np.argmax(X, axis=-1)
        return y


def bandpass(sig, freq0, freq1, srate, axis=-1):
    """带通 + 50Hz 陷波滤波器"""
    wn1 = 2 * freq0 / srate
    wn2 = 2 * freq1 / srate
    b, a = signal.butter(4, [wn1, wn2], 'bandpass')
    srate_ = 1000
    f0 = 50
    Q = 30
    b_notch, a_notch = signal.iirnotch(f0, Q, fs=srate_)
    sig_new = signal.filtfilt(b_notch, a_notch, sig)
    sig_new = signal.filtfilt(b, a, sig, axis=axis)
    return sig_new


def resample_data(X, srate_up, srate_down):
    """重采样"""
    X = np.reshape(X, (-1, X.shape[-2], X.shape[-1]))
    X = resample(X, up=srate_up, down=srate_down)


# ============================================================================
# 模型加载（从离线训练产出的 .pkl 文件）
# ============================================================================

def load_trca_model(model_path):
    """
    从离线训练脚本产出的 .pkl 文件加载 TRCA 模型。
    如果文件不存在，会列出 models/ 目录下已有的模型文件。
    """
    if not os.path.exists(model_path):
        # 列出可用的模型文件
        model_dir = os.path.dirname(model_path) or '.'
        if os.path.isdir(model_dir):
            available = [f for f in os.listdir(model_dir) if f.endswith('.pkl')]
        else:
            available = []
        msg = f"\n❌ 模型文件不存在: {model_path}\n"
        if available:
            msg += f"   {model_dir}/ 目录下已有的模型文件:\n"
            for f in sorted(available):
                msg += f"     • {f}\n"
        else:
            msg += (f"   {model_dir}/ 目录下没有任何 .pkl 文件。\n"
                    f"   请先运行: python offline_TRCA_train.py\n")
        raise FileNotFoundError(msg)

    with open(model_path, 'rb') as f:
        bundle = pickle.load(f)
    model = bundle['model']
    config = bundle['config']

    print(f"[load] 模型已加载: {model_path}")
    print(f"[load] 被试: {config.get('subject', '?')}, "
          f"训练日期: {config.get('train_date', '?')}")
    print(f"[load] 配置: 采样率={config['srate']}Hz, "
          f"窗口={config['t_window']}s, "
          f"类别数={config['n_classes']}, "
          f"导联数={config['n_channels']}")
    return model, config


# ============================================================================
# 放大器驱动（保持不变）
# ============================================================================

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


# ============================================================================
# 在线反馈 Worker（核心修改）
# ============================================================================

class FeedbackWorker(ProcessWorker):
    """
    在线 TRCA 分类 Worker。

    工作流程：
      pre()     → 加载 BDF 校准数据 → 切 epoch → 训练 TRCA 模型
      consume() → 接收实时 epoch → 陷波 → TRCA 预测 → LSL 输出
    """

    def __init__(self, model_path, stim_interval, event_map,
                 srate, timeout, worker_name, ch_ind,
                 freq_list=None, trigger_ch_name=None,
                 feedback_addr=(FEEDBACK_IP, FEEDBACK_PORT),
                 ctl_addr=(CTL_IP, CTL_PORT)):
        """
        参数
        ----
        model_path : str
            离线训练产出的模型文件路径（trca_model.pkl）
            ★ 由 offline_TRCA_train.py 生成 ★
        stim_interval : list
            epoch 时间窗 [tmin, tmax]，单位秒。建议 [0.14, 0.64] 即 0.5s 窗口
        event_map : dict
            事件标签映射（仅在 trigger_ch_name=None 时用于 annotations 提取）
        srate : int
            采样率，默认 1000Hz
        feedback_addr : tuple, optional
            UDP 反馈发送目标地址 (ip, port)，默认发给无人机/界面
        ctl_addr : tuple, optional
            UDP 回环控制目标地址 (ip, port)，主程序据此模拟键盘按键
        timeout : float
            Worker 超时时间
        worker_name : str
            Worker 名称
        ch_ind : array-like
            在线数据的导联索引（从 TCP 收到的 32 通道中选择）
            ★ 必须与离线训练时的导联一致 ★
        freq_list : list, optional
            刺激频率列表，默认 [8, 9, 10, 11, 12, 13] Hz
        trigger_ch_name : str or None
            BDF 触发通道名称（如 "Trigger/Status"）。
            None → 从文件标注提取；字符串 → 从命名通道解析
        """
        self.model_path = model_path
        self.stim_interval = stim_interval
        self.stim_labels = event_map
        self.srate = srate
        self.ch_ind = ch_ind
        # 刺激频率
        self.freq_list = freq_list if freq_list is not None else [8, 9, 10, 11, 12, 13]
        # 历史预测记录（用于计算运行准确率）
        self.pred_history = []
        # 模型配置（加载后填充）
        self.model_config = None
        # 触发通道名称
        self.trigger_ch_name = trigger_ch_name
        # UDP 反馈地址与回环控制地址（socket 在 pre() 中创建）
        self.feedback_addr = feedback_addr
        self.ctl_addr = ctl_addr
        self.sock_feedback = None
        self.sock_ctl = None
        super().__init__(timeout=timeout, name=worker_name)
    # ------------------------------------------------------------------
    # Epoch 切分
    # ------------------------------------------------------------------

    def segment_epoch(self, data, event, tmin=0.0, tmax=1.0, fs=1000):
        """
        根据 event 位置从连续数据中切出 epochs。

        返回
        ----
        epochs : ndarray, shape (n_event, n_chan, n_tpoint)
        """
        n_event = event.shape[0]
        n_chan = data.shape[0]
        n_tpoint = int(fs * tmax - fs * tmin)
        epochs = np.zeros((n_event, n_chan, n_tpoint))
        for idx_event in range(n_event):
            idx_t0 = event[idx_event, 0] + int(fs * tmin)
            idx_t1 = event[idx_event, 0] + int(fs * tmax)
            epochs[idx_event] = data[:, idx_t0:idx_t1]
        return epochs

    # ------------------------------------------------------------------
    # 读取校准数据
    # ------------------------------------------------------------------

    def read_data(self, run_files, train_ch_ind, interval, event_map, fs):
        """
        读取 BDF/CNT 校准文件，切出训练用 epochs。

        返回
        ----
        epoch_raw : ndarray, shape (n_trials, n_chans, n_tpoints)
        labels : ndarray, shape (n_trials,)
        """
        epoch_raw = []
        labels = []
        for idx_bdf in range(len(run_files)):
            fpath = run_files[idx_bdf]
            ext = fpath.lower()

            # 根据文件格式和 trigger_ch_name 决定加载方式
            if ext.endswith('.cnt'):
                raw = mne.io.read_raw_cnt(fpath, preload=True)
            elif ext.endswith('.bdf') and self.trigger_ch_name is not None:
                raw = read_raw_bdf(fpath, trigger_ch_name=self.trigger_ch_name)
            elif ext.endswith('.bdf'):
                raw = mne.io.read_raw_bdf(fpath, preload=True)
            else:
                raw = mne.io.read_raw_bdf(fpath, preload=True)

            data = raw.get_data()
            print(f"原始数据 shape: {data.shape}")

            # ---- 导联选择 ----
            data = data[train_ch_ind, :]
            print(f"选导联后 shape: {data.shape}  (选中 {len(train_ch_ind)} 导联)")

            # 提取事件
            event = extract_events(raw, trigger_ch_name=self.trigger_ch_name)
            labels.append(event[:, -1] - 1)   # 标签从 0 开始

            # 切 epoch
            epoch_raw.append(
                self.segment_epoch(data.copy(), event,
                                   tmin=interval[0], tmax=interval[1], fs=fs)
            )

        epoch_raw = np.concatenate(epoch_raw, axis=0)
        labels = np.concatenate(labels, axis=0)

        print(f"训练数据: epochs {epoch_raw.shape}, 标签范围 [{labels.min()}-{labels.max()}]")
        return epoch_raw, labels

    # ------------------------------------------------------------------
    # 预处理：初始化 + 加载预训练模型
    # ------------------------------------------------------------------

    def pre(self):
        """
        系统启动时执行一次：
          1. 从 .pkl 文件加载离线训练好的 TRCA 模型
          2. 建立 LSL 连接

        ★ 不再从这里训练！模型由 offline_TRCA_train.py 训练并导出 ★
        """
        # ---- Step 1: 加载预训练模型 ----
        print(f"[pre] 加载预训练模型: {self.model_path}")
        self.model, self.model_config = load_trca_model(self.model_path)

        # ---- Step 2: 验证参数一致性 ----
        print(f"[pre] 在线参数: 采样率={self.srate}Hz, "
              f"窗口={self.stim_interval[1]-self.stim_interval[0]:.1f}s, "
              f"导联数={len(self.ch_ind)}")
        print(f"[pre] 模型参数: 采样率={self.model_config['srate']}Hz, "
              f"窗={self.model_config['t_window']}s, "
              f"导联数={self.model_config['n_channels']}, "
              f"类别数={self.model_config['n_classes']}")
        if self.model_config['srate'] != self.srate:
            print(f"  ⚠ 警告：在线采样率({self.srate})与训练时({self.model_config['srate']})不一致！")
        if abs(self.model_config['t_window'] - (self.stim_interval[1]-self.stim_interval[0])) > 0.01:
            print(f"  ⚠ 警告：在线窗口({self.stim_interval[1]-self.stim_interval[0]:.1f}s)"
                  f"与训练时({self.model_config['t_window']:.1f}s)不一致！")
        # ★ 导联数检查：这是最容易出错的地方 ★
        if len(self.ch_ind) != self.model_config['n_channels']:
            raise ValueError(
                f"❌ 导联数不匹配！\n"
                f"   在线 ch_ind 选了 {len(self.ch_ind)} 个导联\n"
                f"   离线模型训练了 {self.model_config['n_channels']} 个导联\n"
                f"   请修改 ch_ind 使其选出 {self.model_config['n_channels']} 个导联")

        # ---- Step 2.5: 预计算陷波滤波器系数（性能优化）----
        # 系数只和采样率有关，与数据无关，在这里算一次即可，
        # 避免每次 consume 都重新调用 iirnotch（消费端提速，减轻 CPU 抢占）。
        self._notch_b, self._notch_a = signal.iirnotch(50, 30, fs=self.srate)

        # ---- Step 3: 建立 UDP 输出 socket（反馈 + 回环控制）----
        self.sock_feedback = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock_ctl = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        print('Connected. 在线分类就绪。')

    # ------------------------------------------------------------------
    # 在线处理：每收到一个 epoch 调用一次
    # ------------------------------------------------------------------

    def consume(self, data):
        """
        在线分类核心——每 0.5 秒触发一次。

        数据流：
          实时数据 (list) → numpy 转置 → 选导联 → 50Hz 陷波
          → TRCA 预测 → 输出标签

        参数
        ----
        data : list
            从 RingBuffer 传来的一个 epoch 原始数据
            shape: (n_samples, n_channels)
        """
        # ---- Step 1: 数据格式转换 ----
        # data 进来是 list of lists: (n_samples, n_channels)
        data = np.array(data, dtype=np.float64).T   # → (n_channels, n_samples)
        print(f"[consume] 原始数据 shape: {data.shape}")

        # ---- Step 2: 导联选择 ----
        data = data[self.ch_ind]   # → (n_selected_chans, n_samples)

        # ---- Step 3: 50Hz 陷波滤波（用 pre() 预计算好的系数，不再重算）----
        data = signal.filtfilt(self._notch_b, self._notch_a, data)

        # ---- Step 4: TRCA 预测 ----
        # TRCA 需要 3D 输入： (n_trials, n_chans, n_samples)
        # 单个试次 → 添加 batch 维度
        data_3d = data[np.newaxis, :, :]   # (1, n_chans, n_samples)
        p_labels = self.model.predict(data_3d)
        # 模型 classes_ 本就是 1~6（离线训练时标签保留原始 1-based），
        # predict() 直接返回 1~6，无需再 +1。
        pred_label = int(p_labels[0])-1

        # 防御：万一标签超出 freq_list 范围，安全兜底不崩溃
        if 1 <= pred_label <= len(self.freq_list):
            _freq_str = f"{self.freq_list[pred_label - 1]} Hz"
        else:
            _freq_str = "未知(标签超范围)"
        print(f"[consume] TRCA 预测结果: {pred_label}  (频率 {_freq_str})")

        # ---- Step 5: 运行准确率统计 ----
        self.pred_history.append(pred_label)
        if len(self.pred_history) > 1:
            # 统计各标签的出现频次
            unique, counts = np.unique(self.pred_history, return_counts=True)
            most_common = unique[np.argmax(counts)]
            consistency = counts.max() / counts.sum()
            print(f"   历史预测: {len(self.pred_history)} 次 "
                  f"| 多数标签: {most_common} | 一致性: {consistency:.1%}")

        # ---- Step 6: UDP 输出 ----
        now = datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3]
        msg = bytes(str(pred_label), encoding='utf-8')
        # 发送给无人机/界面
        self.sock_feedback.sendto(msg, self.feedback_addr)
        # 回环发给主进程，用于键盘控制
        self.sock_ctl.sendto(msg, self.ctl_addr)

        print("p_labels send done at {}".format(now))
        print("********************************************\r\n")

    # ------------------------------------------------------------------
    # 清理
    # ------------------------------------------------------------------

    def post(self):
        """Worker 停止时调用，关闭 UDP socket"""
        for sock in (self.sock_feedback, self.sock_ctl):
            try:
                sock.close()
            except Exception:
                pass


# ============================================================================
# 主程序
# ============================================================================

if __name__ == '__main__':

    # 打印当前生效的共享配置（导联/窗长），方便核对是否与离线训练一致
    print_config()

    # ======================== ⚙ 配置参数（按需修改） ========================

    # 放大器采样率
    srate = 1000

    # 分类时间窗从共享配置读取（改窗长请去 config_channels.py 的 CLASSIFY_WINDOW）
    stim_interval = STIM_INTERVAL

    # 刺激频率（Hz）—— 必须与实际视觉刺激一致
    freq_list = [9.6, 11.6, 9.2,12.8, 10.4, 8.8, 13.2, 10.8, 10.0, 11.2, 12.0, 12.4]
    # 标签配置
    stim_labels = list(range(1, len(freq_list) + 1))
    event_map = {str(e): e for e in range(1, 255)}

    # ======================== 模型文件路径 ========================
    # 默认自动加载 models/ 目录下【最新】生成的 .pkl 模型（按修改时间）。
    # 离线训练完直接跑在线即可，不用手动改文件名。
    _model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')

    _pkls = [f for f in os.listdir(_model_dir) if f.endswith('.pkl')]
    if not _pkls:
        raise FileNotFoundError(f"{_model_dir} 下没有 .pkl 模型，请先运行 offline_TRCA_train.py")
    _latest = max(_pkls, key=lambda f: os.path.getmtime(os.path.join(_model_dir, f)))
    model_path = os.path.join(_model_dir, _latest)
    
    # ★ 若要指定用某个模型（而非最新），把上面注释掉，手动写死路径即可：
    # model_path = os.path.join(_model_dir, 'trca_model_djm_20260723_235802.pkl')
    
    # ======================== 导联索引（在线） ========================
    # 从共享配置读取有效导联索引（已自动剔除坏导联，顺序与离线训练一致）。
    # ★ 要增删坏导联，请去 config_channels.py 改 BAD_CHANNELS，不要改这里 ★
    ch_ind = GOOD_CH_INDEX

    # ======================== ★ 事件提取方式 ★ ========================
    # None → 从文件标注 (annotations) 提取事件
    # "Trigger/Status" → 从 BDF 命名触发通道解析事件
    trigger_ch_name = None

    # ======================== Worker 配置 ========================
    feedback_worker_name = 'feedback_worker'

    # ======================== 放大器配置 ========================
    amp_ip = '127.0.0.1'    # NeuroScan TCP 数据转发地址
    amp_port = 1895          # NeuroScan TCP 数据转发端口
    amp_chans = 64           # 放大器在线传输的总通道数

    # ======================== 构建系统 ========================
    # UDP 反馈 socket 由 FeedbackWorker 在 pre() 中自行创建，这里无需手动创建。
    worker = FeedbackWorker(
        model_path=model_path,
        stim_interval=stim_interval,
        event_map=event_map,
        srate=srate,
        timeout=5e-2,
        worker_name=feedback_worker_name,
        ch_ind=ch_ind,
        freq_list=freq_list,
        trigger_ch_name=trigger_ch_name,
    )

    marker = Marker(stim_interval, srate, stim_labels)

    # ======================== 启动在线系统 ========================
    ns = DitingBrainEEGAmplifier(
        device_address=(amp_ip, amp_port),
        srate=srate,
        num_chans=amp_chans
    )
    ns.register_worker(feedback_worker_name, worker, marker)
    ns.up_worker(feedback_worker_name)
    time.sleep(0.5)
    ns.start_trans()

    # ======================== 接收回环指令（用于键盘控制）============================
    sock_ctl_recv = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock_ctl_recv.bind((CTL_IP, CTL_PORT))

    '''
    键盘按键相关操作----------------------------------------------------
    '''
    keyboard = Controller()

    # 12 个指令 → 键盘按键 映射（先按此方案实现，后续可再调整）
    # 0 TakeOff / 1 Land / 2 Forward / 3 Back / 4 Left / 5 Right
    # 6 CW / 7 CCW / 8 Up / 9 Down / 10 Keep / 11 Hover
    command_key = {
        0:  Key.space,                 # TakeOff 起飞
        1:  KeyCode.from_char('l'),    # Land 降落
        2:  Key.up,                    # Forward 前进
        3:  Key.down,                  # Back 后退
        4:  Key.left,                  # Left 左移
        5:  Key.right,                 # Right 右移
        6:  KeyCode.from_char('d'),    # CW 顺时针旋转
        7:  KeyCode.from_char('a'),    # CCW 逆时针旋转
        8:  KeyCode.from_char('w'),    # Up 上升
        9:  KeyCode.from_char('s'),    # Down 下降
        10: KeyCode.from_char('k'),    # Keep 保持
        11: KeyCode.from_char('h'),    # Hover 悬停
    }

    print("键盘控制已就绪，等待指令... (Ctrl+C 退出)")

    CMD_KEEP = 10    # Keep：保持
    CMD_HOVER = 11   # Hover：悬停（只松键）

    current_key = None   # 当前正在按下的按键（收到下一条命令时才松开）
    try:
        while True:
            recv_data = sock_ctl_recv.recvfrom(1024)
            data = int(recv_data[0].decode('utf-8'))
            print("指令：", data)

            # Keep：保持当前按键不松开，也不按新键
            if data == CMD_KEEP:
                continue

            # Hover：只松开当前按键，不按下新键
            if data == CMD_HOVER:
                if current_key is not None:
                    keyboard.release(current_key)
                    current_key = None
                continue

            key = command_key.get(data)
            if key is None:
                print("未知指令，忽略:", data)
                continue

            # 与当前按键相同则继续按住，不重复触发
            if key == current_key:
                continue

            # 先松开上一个按键，再按下新按键
            if current_key is not None:
                keyboard.release(current_key)
            keyboard.press(key)
            current_key = key
    except KeyboardInterrupt:
        print("\n停止接收指令")
        if current_key is not None:
            keyboard.release(current_key)

    ns.down_worker('feedback_worker')
    time.sleep(1)
    ns.stop_trans()
    ns.clear()
    print('bye')
