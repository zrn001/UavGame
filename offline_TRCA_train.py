# -*- coding: utf-8 -*-
"""
离线 TRCA 训练与评估脚本。

功能：
  1. 支持 .cnt 和 .bdf 两种文件格式（自动识别）
  2. 通过 k 折交叉验证评估 TRCA/FBTRCA 分类准确率
  3. 输出详细的每折准确率、混淆矩阵、平均准确率

基于：
  - HardwareAmplifierTesting.py 的离线处理架构
  - TRCA_BrainCar.py 的模型参数配置

用法：
  1. 修改 CONFIG 部分的 data_files 指向你的数据文件
  2. python offline_TRCA_train.py
"""

import os
import sys
import pickle
from datetime import datetime
import numpy as np
from scipy import signal
import mne

# 让 print 的中文/emoji 在 Windows GBK 终端也能正常输出，不再因编码崩溃
try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass
from metabci.brainda.algorithms.decomposition import FBTRCA
from metabci.brainda.algorithms.decomposition.base import (
    generate_filterbank, generate_cca_references)
from utils.extract_events import extract_events, read_raw_bdf
# ★ 从共享配置读取：坏导联、分类窗长（改配置只需改 config_channels.py）
from config_channels import GOOD_CHANNELS, STIM_INTERVAL, print_config

mne.set_log_level('WARNING')


# ============================================================================
# ① 文件加载：自动识别 .cnt / .bdf 格式
# ============================================================================

def load_raw_file(file_path, trigger_ch_name=None):
    """
    根据文件后缀自动选择读取方式。

    参数
    ----
    file_path : str
        文件路径，支持 .cnt / .bdf / .edf 格式
    trigger_ch_name : str or None
        BDF 文件中的触发通道名称（如 "Trigger/Status"）。
        - None → 从文件标注 (annotations) 提取事件（兼容 .cnt 及有标注的文件）
        - 字符串 → BDF 文件用 mne.find_events(stim_channel=...) 从该通道解析事件

    返回
    ----
    raw : mne.io.Raw
        MNE Raw 对象
    events : ndarray, shape (n_events, 3)
        MNE 格式的事件数组 [sample, 0, label]
    ch_names : list
        导联名称列表
    """
    ext = os.path.splitext(file_path)[1].lower()

    if ext == '.cnt':
        print(f"  [CNT] 读取: {file_path}")
        if trigger_ch_name is not None:
            print(f"  ⚠ 警告: .cnt 文件不支持 stim_channel，将忽略 trigger_ch_name"
                  f"并回退到标注模式")
        raw = mne.io.read_raw_cnt(file_path, preload=True)
    elif ext == '.bdf':
        print(f"  [BDF] 读取: {file_path}")
        if trigger_ch_name is not None:
            # BDF + 触发通道 → 用 MNE 原生 stim_channel 加载
            print(f"        使用触发通道: '{trigger_ch_name}'")
            raw = read_raw_bdf(file_path, trigger_ch_name=trigger_ch_name)
        else:
            raw = mne.io.read_raw_bdf(file_path, preload=True)
    elif ext == '.edf':
        print(f"  [EDF] 读取: {file_path}")
        raw = mne.io.read_raw_edf(file_path, preload=True)
    else:
        raise ValueError(f"不支持的文件格式: '{ext}'。仅支持 .cnt / .bdf / .edf")

    ch_names = raw.ch_names

    # ---- 提取事件（双模式：触发通道 或 文件标注） ----
    events = extract_events(raw, trigger_ch_name=trigger_ch_name)

    print(f"        导联数: {len(ch_names)}, 采样率: {raw.info['sfreq']} Hz, "
          f"事件数: {len(events)}, 时长: {raw.times[-1]:.1f}s")

    return raw, events, ch_names


# ============================================================================
# ② 预处理
# ============================================================================

class Preprocessor:
    """
    脑电数据预处理流水线。

    处理顺序：去导联 → 重参考 → 选导联 → 带通滤波 → 重采样
    """

    def __init__(self, ch_names, fs, drop_channels=None, refer=None,
                 chan_sel=None, passband=None, resample_srate=None):
        self.ch_names = ch_names
        self.fs = fs
        self.drop_channels = drop_channels
        self.refer = refer
        self.chan_sel = chan_sel
        self.passband = passband
        self.resample_srate = resample_srate

    def process(self, raw):
        """对 Raw 对象执行预处理，返回 numpy 数组 (n_channels, n_times)"""
        # ① 删除不需要的导联
        if self.drop_channels:
            raw.drop_channels(self.drop_channels)

        # ② 重参考
        if self.refer:
            raw.set_eeg_reference(ref_channels=self.refer)

        # ③ 导联选择
        # 先 pick 选出目标导联，再 reorder 强制顺序与 chan_sel 完全一致。
        # （MNE 不同版本 pick_channels 的顺序行为不一致，reorder 是关键保险，
        #  确保离线导联顺序 == 在线 GOOD_CH_INDEX 顺序，否则 TRCA 会学错。）
        if self.chan_sel:
            raw.pick_channels(self.chan_sel)
            raw.reorder_channels(self.chan_sel)

        # ④ 带通滤波
        if self.passband:
            raw.filter(l_freq=self.passband[0], h_freq=self.passband[1],
                       method='iir')

        # ⑤ 重采样
        if self.resample_srate and self.resample_srate != self.fs:
            raw.resample(self.resample_srate)

        return raw.get_data()


# ============================================================================
# ③ Epoch 切分
# ============================================================================

def segment_epochs(data, events, fs, tmin=0.0, tmax=1.0):
    """
    根据事件位置切分连续脑电数据。

    参数
    ----
    data : ndarray, shape (n_channels, n_times)
    events : ndarray, shape (n_events, 3)
        MNE 事件数组
    fs : int
        采样率
    tmin : float
        事件后开始截取的时间（秒），默认 0（从事件那一刻开始）
    tmax : float
        事件后结束截取的时间（秒），默认 1.0

    返回
    ----
    epochs : ndarray, shape (n_events, n_channels, n_tpoints)
    labels : ndarray, shape (n_events,)
    """
    n_events = events.shape[0]
    n_chans = data.shape[0]
    n_tpoints = int(fs * tmax - fs * tmin)
    epochs = np.zeros((n_events, n_chans, n_tpoints))

    for i in range(n_events):
        idx_t0 = events[i, 0] + int(fs * tmin)
        idx_t1 = events[i, 0] + int(fs * tmax)
        epochs[i] = data[:, idx_t0:idx_t1]

    labels = events[:, 2].copy()

    print(f"  切分完成: {n_events} 个 epochs, "
          f"shape {epochs.shape}, "
          f"标签范围 [{labels.min()}-{labels.max()}]")
    print(f"  标签分布: {dict(zip(*np.unique(labels, return_counts=True)))}")

    return epochs, labels


# ============================================================================
# ④ 去基线 + 标准化（可选）
# ============================================================================

def baseline_correction(X):
    """去基线 + Z-score 标准化"""
    X = np.reshape(X, (-1, X.shape[-2], X.shape[-1]))
    X = X - np.mean(X, axis=-1, keepdims=True)
    X = X / np.std(X, axis=(-1, -2), keepdims=True)
    return X


# ============================================================================
# ⑤ k 折交叉验证（类别平衡）
# ============================================================================

def leave_one_block_cv(data, labels, model, n_folds=6, notch_fs=1000):
    """
    类别平衡的 k 折交叉验证。

    将每个类别的样本均匀分成 n_folds 份，
    每次取 1 份做测试、其余做训练，确保各类别在每折的比例相同。

    参数
    ----
    data : ndarray, shape (n_trials, n_chans, n_tpoints)
    labels : ndarray, shape (n_trials,)
    model : sklearn-like estimator
        必须有 fit() 和 predict() 方法
    n_folds : int
        交叉验证折数
    notch_fs : int
        陷波滤波器采样率（0 表示不做陷波）

    返回
    ----
    results : dict
        'accuracies'   : 每折准确率
        'y_true'       : 真实标签（全部分）
        'y_pred'       : 预测标签（全部分）
        'mean_acc'     : 平均准确率
        'std_acc'      : 准确率标准差
    """
    unique_labels = np.unique(labels)
    n_classes = len(unique_labels)

    # ---- 50Hz 陷波滤波 ----
    if notch_fs > 0:
        f0, Q = 50, 30
        b_notch, a_notch = signal.iirnotch(f0, Q, fs=notch_fs)
        # 对每个试次做零相位滤波
        for i in range(data.shape[0]):
            data[i] = signal.filtfilt(b_notch, a_notch, data[i])

    # 计算每个类别每折的试次数
    # 确保所有类别的每折样本数相同（平衡）
    n_per_class = min(np.sum(labels == l) for l in unique_labels)
    n_per_fold = n_per_class // n_folds
    if n_per_fold < 1:
        raise ValueError(f"每类样本数({n_per_class})不足以做{n_folds}折交叉验证")

    print(f"\n  每类样本数: {n_per_class}, 每折每类: {n_per_fold}, 折数: {n_folds}")

    accs, y_trues, y_preds = [], [], []

    for fold in range(n_folds):
        # ---- 划分训练/测试集（每类平衡采样） ----
        idx_test_list, idx_train_list = [], []

        for l in unique_labels:
            class_idx = np.where(labels == l)[0]
            # 只使用前 n_per_class 个样本
            class_idx = class_idx[:n_per_class]
            np.random.shuffle(class_idx)

            start = fold * n_per_fold
            end = (fold + 1) * n_per_fold
            idx_test = class_idx[start:end]
            idx_train = np.setdiff1d(class_idx, idx_test)

            idx_test_list.append(idx_test)
            idx_train_list.append(idx_train)

        idx_test = np.concatenate(idx_test_list)
        idx_train = np.concatenate(idx_train_list)

        # ---- 训练 & 预测 ----
        X_train, y_train = data[idx_train], labels[idx_train]
        X_test, y_test = data[idx_test], labels[idx_test]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = np.mean(y_pred == y_test)
        accs.append(acc)
        y_trues.append(y_test)
        y_preds.append(y_pred)

        print(f"    Fold {fold+1}/{n_folds}: 准确率 = {acc:.4f}  "
              f"(训练:{len(idx_train)}, 测试:{len(idx_test)})")

    # ---- 汇总结果 ----
    accs = np.array(accs)
    y_true_all = np.concatenate(y_trues)
    y_pred_all = np.concatenate(y_preds)

    return {
        'accuracies': accs,
        'y_true': y_true_all,
        'y_pred': y_pred_all,
        'mean_acc': accs.mean(),
        'std_acc': accs.std(),
    }


# ============================================================================
# ⑥ 结果报告
# ============================================================================

def print_report(results, n_folds):
    """打印详细评估报告"""
    print("\n" + "=" * 60)
    print("                    离 线 评 估 报 告")
    print("=" * 60)

    # 整体准确率
    print(f"\n  📊 整体准确率: {results['mean_acc']:.4f} ± {results['std_acc']:.4f} "
          f"({n_folds} 折交叉验证)")
    print(f"  各折准确率: ", end="")
    for i, acc in enumerate(results['accuracies']):
        print(f"Fold{i+1}={acc:.4f}  ", end="")
    print()

    # 整体混淆矩阵
    y_true = results['y_true']
    y_pred = results['y_pred']
    labels_uniq = np.unique(y_true)
    print(f"\n  📋 混淆矩阵 (行=真实, 列=预测):")
    header = "        " + "".join(f"  Pred{int(l):2d}" for l in labels_uniq)
    print(header)
    for true_l in labels_uniq:
        row = f"  True{int(true_l):2d}:"
        for pred_l in labels_uniq:
            count = np.sum((y_true == true_l) & (y_pred == pred_l))
            row += f"    {count:3d}"
        print(row)

    # 各类别准确率
    print(f"\n  🎯 各类别准确率:")
    for l in labels_uniq:
        mask = y_true == l
        class_acc = np.mean(y_pred[mask] == l)
        n_samples = np.sum(mask)
        print(f"    类别 {int(l)}: {class_acc:.4f}  (n={n_samples})")

    print("\n" + "=" * 60)


# ============================================================================
# ⑦ 模型构建
# ============================================================================

def build_fbtrca_model(wp, ws, srate, n_components=1, ensemble=True):
    """
    构建 FBTRCA 模型。

    参数
    ----
    wp : list of tuple
        滤波器组通带 [(low1,high1), (low2,high2), ...]
    ws : list of tuple
        滤波器组阻带
    srate : int
        采样率
    n_components : int
        TRCA 成分数（默认 1）
    ensemble : bool
        是否使用集成模式

    返回
    ----
    model : FBTRCA
    """
    filterbank = generate_filterbank(wp, ws, srate=srate)
    filterweights = np.arange(1, len(wp) + 1) ** (-1.25) + 0.25

    model = FBTRCA(
        filterbank=filterbank,
        n_components=n_components,
        ensemble=ensemble,
        filterweights=filterweights,
        n_jobs=-1
    )
    return model


def build_trca_model(n_components=2):
    """构建基础 TRCA 模型（无滤波器组）"""
    return FBTRCA(n_components=n_components)


# ============================================================================
# ⑧ 模型保存与加载（离线 → 在线的桥梁）
# ============================================================================

def save_model(model, config, save_path):
    """保存训练好的模型和配置到 .pkl 文件"""
    bundle = {'model': model, 'config': config}
    with open(save_path, 'wb') as f:
        pickle.dump(bundle, f)
    print(f"\n  💾 模型已保存至: {save_path}")
    print(f"     文件大小: {os.path.getsize(save_path) / 1024:.1f} KB")


def load_model(load_path):
    """从 .pkl 文件加载模型"""
    with open(load_path, 'rb') as f:
        bundle = pickle.load(f)
    return bundle['model'], bundle['config']


# ============================================================================
# ⑨ 主程序
# ============================================================================

if __name__ == '__main__':

    # ======================== ⚙ 配置参数（按需修改） ========================

    # ---- 被试信息 & 模型保存 ----
    subject = 'ltf'                         # ★ 被试姓名简称，用于模型文件命名
    model_dir = './models'                  # ★ 模型保存目录（自动创建）
    # 文件名精确到 时分秒，同一天多次训练不会互相覆盖
    model_name = f'trca_model_{subject}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pkl'
    # ↑ 例: trca_model_djm_20260723_184257.pkl

    # ---- 数据文件（支持 .cnt / .bdf / .edf，可混合） ----
    data_files = [r"D:\Desktop\uavdata\ltf_20260918_offline_1.bdf"
                  # r"C:\Users\djm\Documents\DitingBrain\EEG_20260722_17_15_59.bdf"
                  # r"C:\Users\djm\Documents\DitingBrain\EEG_20260721_22_43_41.bdf"
                  # r'D:\脑控小车\脑控小车djm\code\Process-Nano\Data\Tju-testOffline\1130\S2.bdf',
                  # r'C:\Users\DELL\Desktop\移动\0408\offline2.bdf',
                  # r'C:\Users\djm\Desktop\6-HardwareAmplifierTesting\...\data\ssvep-offline3.cnt',
    ]

    # ---- 采样率 & 时间窗 ----
    srate = 1000          # 放大器采样率 (Hz)
    # 时间窗从共享配置读取（改窗长请去 config_channels.py 的 CLASSIFY_WINDOW）
    tmin = STIM_INTERVAL[0]   # 视觉延迟补偿 (秒)
    tmax = STIM_INTERVAL[1]   # 数据窗结束 (秒) = tmin + 分类窗长

    # ---- 带通滤波 ----
    l_freq, h_freq = 1, 90

    # ---- 滤波器组（FBTRCA 专用） ----
    # 子带数 = n_classes，覆盖 SSVEP 基频和谐波
    wp = [(5, 90), (14, 90), (22, 90), (30, 90)]
    ws = [(3, 92), (12, 92), (20, 92), (28, 92)]

    # ---- TRCA 参数 ----
    n_components = 1      # 每类保留 1 个 TRCA 成分

    # ---- 交叉验证 ----
    n_folds = 4           # k 折数

    # ---- 预处理 ----
    do_baseline_corr = False
    do_notch = True

    # ---- 导联选择 ----
    # 从共享配置读取有效导联（已自动剔除坏导联）。
    # ★ 要增删坏导联，请去 config_channels.py 改 BAD_CHANNELS，不要改这里 ★
    chan_sel = GOOD_CHANNELS

    # ---- ★ 事件提取方式 ★ ----
    # BDF 文件中触发通道的名称（如 "Trigger/Status"）。
    # 设为 None → 从文件标注 (annotations) 提取事件
    # 设为字符串 → 从该命名通道的信号中解析事件（适配无标注的 BDF 文件）
    trigger_ch_name = "Trigger/Status"   # None=从标注提取, "Trigger/Status"=从触发通道提取

    # ---- 模型选择 ----
    model_type = 'FBTRCA'

    # ======================== 主流程 ========================

    # 打印当前生效的共享配置（导联/窗长），方便核对
    print_config()

    print("=" * 60)
    print("        离 线 TRCA 训 练 与 评 估")
    print("=" * 60)
    print(f"  数据文件数: {len(data_files)}")
    print(f"  时间窗口: [{tmin}, {tmax}] = {tmax-tmin:.1f}s")
    print(f"  带通滤波: {l_freq}-{h_freq} Hz")
    print(f"  模型: {model_type}, n_components={n_components}")
    print(f"  交叉验证: {n_folds} 折")
    print("-" * 60)

    # ---- Step 1: 加载所有数据文件 ----
    print("\n[1/5] 加载数据文件...")
    X_list, y_list = [], []

    for fp in data_files:
        if not os.path.exists(fp):
            print(f"  ⚠ 跳过不存在的文件: {fp}")
            continue

        raw, events, raw_ch_names = load_raw_file(fp, trigger_ch_name=trigger_ch_name)

        # ---- 预处理 ----
        preprocessor = Preprocessor(
            ch_names=raw_ch_names,    # 文件中全部导联列表（仅用于创建 Info）
            fs=srate,
            drop_channels=None,       # 不需要的导联，如 ['HEOG', 'VEOG']
            refer=None,               # 重参考，如 ['TP9', 'TP10'] 做乳突参考
            chan_sel=chan_sel,        # ★ 用配置区定义的导联做选择
            passband=[l_freq, h_freq],
            resample_srate=srate if raw.info['sfreq'] != srate else None
        )
        data = preprocessor.process(raw)

        # ---- 切分 Epochs ----
        epochs, labels = segment_epochs(data, events, fs=srate, tmin=tmin, tmax=tmax)
        X_list.append(epochs)
        y_list.append(labels)

    if not X_list:
        print("\n❌ 没有找到有效的数据文件！请在 CONFIG 中设置 data_files。")
        sys.exit(1)

    # ---- 合并所有文件的数据 ----
    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    print(f"\n  合并后总数据: {X.shape}, 标签: {y.shape}")

    # ---- Step 2: 可选基线校正 ----
    if do_baseline_corr:
        print("\n[2/5] 基线校正 + 标准化...")
        X = baseline_correction(X)
    else:
        print("\n[2/5] 跳过基线校正")

    # ---- Step 3: 构建模型 ----
    print(f"\n[3/5] 构建 {model_type} 模型...")
    if model_type == 'FBTRCA':
        model = build_fbtrca_model(wp, ws, srate, n_components=n_components)
    elif model_type == 'TRCA':
        model = build_trca_model(n_components=n_components)
    else:
        raise ValueError(f"未知模型类型: {model_type}")

    print(f"  滤波器组: {len(wp)} 个子带")
    for i, (wpi, wsi) in enumerate(zip(wp, ws)):
        print(f"    子带{i+1}: 通带{wpi}, 阻带{wsi}")

    # ---- Step 4: 交叉验证 ----
    print(f"\n[4/5] {n_folds} 折交叉验证...")
    notch_fs = srate if do_notch else 0
    results = leave_one_block_cv(X, y, model, n_folds=n_folds, notch_fs=notch_fs)

    # ---- Step 5: 报告 ----
    print("\n[5/5] 生成报告...")
    print_report(results, n_folds)

    # ---- 保存 CV 结果 ----
    print("✅ 交叉验证评估完成！")
    print(f"   平均准确率: {results['mean_acc']:.4f} (±{results['std_acc']:.4f})")

    # ==================== Step 6: 全量训练 & 导出模型 ====================
    print("\n" + "-" * 60)
    print("[6/6] 使用全部数据训练最终模型 → 导出供在线使用...")

    # 重新构建新模型（不被 CV 中的 fit 污染）
    if model_type == 'FBTRCA':
        final_model = build_fbtrca_model(wp, ws, srate, n_components=n_components)
    else:
        final_model = build_trca_model(n_components=n_components)

    # 全部数据做陷波
    if do_notch:
        f0, Q = 50, 30
        b_notch, a_notch = signal.iirnotch(f0, Q, fs=srate)
        X_final = X.copy()
        for i in range(X_final.shape[0]):
            X_final[i] = signal.filtfilt(b_notch, a_notch, X_final[i])
    else:
        X_final = X

    final_model.fit(X_final, y)

    # 打包配置写给在线系统
    config = {
        'subject': subject,
        'train_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'srate': srate,
        'tmin': tmin, 'tmax': tmax,
        't_window': tmax - tmin,
        'wp': wp, 'ws': ws,
        'n_components': n_components,
        'n_channels': X.shape[1],
        'n_classes': len(np.unique(y)),
        'passband': [l_freq, h_freq],
        'do_notch': do_notch,
    }

    # 创建保存目录 + 自动命名
    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), model_dir)
    os.makedirs(save_dir, exist_ok=True)
    model_save_path = os.path.join(save_dir, model_name)
    save_model(final_model, config, model_save_path)
    print(f"\n  ⭐ 在线系统加载路径: {model_save_path}")
