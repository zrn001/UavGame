import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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


# def event_pre():



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


# # '''列建模'''
# # # # .OK now 
# # ---------------------------------------------读取csv格式数据--------------------------------------------
csv_path = "C:\\Users\\Administrator\\Desktop\\1106炳杰数据待分析\\20241106_SSVEP_Data1.csv"
data = pd.read_csv(csv_path)
eeg_data = data.iloc[:, 1:9].values.T

event_times = data.iloc[:, 9].dropna().values.astype(int)  # 第十列为事件采样点时间
event_labels = data.iloc[:, 10].dropna().values.astype(int)  # 第十一列为事件标签
print("eeg_data:",eeg_data)
# print(event_times)
# print(event_labels)
unique_events = []
unique_times = []
 
filtered_times = []   #  标签出现的时间点
filtered_labels = []    # 标签
current_round = set()  # 存储当前轮次已出现的标签
round_count = 0  # 用于记录轮次

n_samples = 1000   # 每个标签后截取的采样点数量
times = 0.2
n_channels = 8     # 通道数量
n_trials = 4       # trils
labels_per_trial = 8  # 每轮有8个不同的标签

# 创建空数组来存储最终数据
final_data = np.zeros((n_trials * labels_per_trial, n_channels, int(n_samples * times)))

for time, label in zip(event_times, event_labels):
    if label == 0:
        pass
    else:
        unique_events.append(label)
        unique_times.append(time)

for time, label in zip(unique_times, unique_events):
    # 判断轮次是否结束（1~8标签应出现一次为一轮）
    if label == 1 and current_round == {1, 2, 3, 4, 5, 6, 7, 8}:  
        current_round.clear()  # 开始新的轮次
        round_count += 1
    
    # 仅保留当前轮次中每个标签的第一次出现
    if label not in current_round:
        filtered_times.append(time)
        filtered_labels.append(label)
        current_round.add(label)
# print(filtered_times)
# print(filtered_labels)

events = np.column_stack((filtered_times, np.zeros(len(filtered_labels), int), filtered_labels))

# 输出检查
print("MNE事件格式的数组：")
print(events)
print(events[:,0])
# for i in range(32):
#     print(events[i+1,0]-events[i,0])

# 从事件中提取每个标签对应的2000个采样点数据
trial_counter = 0  # 用于标记填入第几个 trial
for i, event in enumerate(events):
    sample_idx, _, label = event  # 获取事件的采样点和标签
    # print(sample_idx)
    sample_idx = sample_idx -19173
    # 检查是否超出原始数据长度
    if sample_idx + n_samples <= eeg_data.shape[1]:
        # 截取从当前事件开始的 2000 个数据点
        data_segment = eeg_data[:, sample_idx:sample_idx + int(n_samples * times)]
        # print(data_segment)
        # 将数据段填入 final_data
        final_data[trial_counter, :, :] = data_segment
        trial_counter += 1

    # 检查是否已完成所有32个试验
    if trial_counter >= n_trials * labels_per_trial:
        break

# 检查最终数据形状是否为 (32, 8, 2000)
print("Final data shape:", final_data.shape)  # 应为 (32, 8, 2000)

X_simu = final_data
print(X_simu)

pick_chs = ['CH1', 'CH2', 'CH3', 'CH4', 'CH5', 'CH6', 'CH7', 'CH8']

freq_list = [8, 9, 10, 11, 12, 13, 14, 15]
Yf = generate_cca_references(freq_list, srate=1000, T=2,n_harmonics = 5)

wp = [[6, 88], [14, 88], [22, 88], [30, 88], [38, 88]
]
ws = [[4, 90], [12, 90], [20, 90], [28, 90], [36, 90]
]
filterweights = np.arange(1, 6)**(-1.25) + 0.25
filterbank = generate_filterbank(wp, ws, 1000)

y_simu = np.array([1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8])
print("y_simu:",y_simu)
print('read data successful')
kfold_accs = []
model = FBTRCA(filterbank=filterbank, n_components=5, filterweights=filterweights, n_jobs=-1)

i = 0
y = np.reshape(y_simu, (-1))
print(y_simu.shape)
spliter = EnhancedLeaveOneGroupOut(return_validate=False)       # 留一法交叉验证
for train_ind, test_ind in spliter.split(X_simu, y=y):
    # print("train_ind:",train_ind)
    # print("test_ind:",test_ind)
    X_train, y_train = np.copy(X_simu[train_ind]), np.copy(y[train_ind])
    X_test, y_test = np.copy(X_simu[test_ind]), np.copy(y[test_ind])
    model = model.fit(X_train, y_train,None)          # 训练模型
    p_labels = model.predict(X_test)                  # 预测标签
    kfold_accs.append(np.mean(p_labels == y_test))    # 记录正确率
    print(kfold_accs)
acc_end = np.mean(kfold_accs)
print(acc_end)

