import mne
import numpy as np
import pandas as pd
from scipy import signal, fftpack
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from metabci.brainda.algorithms.decomposition import TRCA
mne.set_log_level(False)


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


def leave_one_block_out_test(data, label, model, n_block):
    """ Leave-one-block-out cross validation test. """
    result = []
    n_epoch = data.shape[0]
    n_epoch_block = int(data.shape[0]/n_block)
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


def get_spectrum(y, fs, nfft=None):
    # FFT
    nfft = len(y) if nfft == None else int(nfft)
    y_fft = fftpack.fft(y, nfft)
    # Frequency interal
    f = np.arange(0, nfft, 1) * fs / nfft
    # Amplitude (normalized) and phase (-pi~pi)
    amp = np.abs(y_fft) / (len(y)/2)
    phase = np.angle(y_fft, deg=False)
    # Single-sided spectrum
    amp, phase, f = amp[:int(nfft/2)], phase[:int(nfft/2)], f[:int(nfft/2)]
    return amp, phase, f


# In[]: Config
event_map = {str(e):e for e in range(1, 9)}
fs = 1000
tmax = 2.14
tmin = 0.0
offset = 0.14
n_chan = 8
freqs = [8, 9, 10, 11, 12, 13, 14, 15]
n_target = len(freqs)


# # In[]: Load data
# ### CSV
epoch1_raw, epoch1_filt, label1 = [], [], []
for idx_csv in range(1, 3):
    # Load
    data1 = pd.read_csv("./demos/brainflow_demos/Data/yyy_offline{}.csv".format(idx_csv))
    data1_raw = data1.iloc[:, 1:9].values.T
    
    # Event
    event1_label = data1.iloc[:, 10].dropna().values.astype(int)
    idx_event1 = np.where(np.diff(event1_label)>0)[0] + 1
    event1 = np.array([idx_event1, np.zeros((len(idx_event1))), event1_label[idx_event1]], dtype=np.int64).T
    label1.append(event1_label[idx_event1]-1)
    
    # Raw for classification
    epoch1_raw.append(segment_epoch(data1_raw.copy(), event1, tmin, tmax, fs))
    
    # Filtering for analysis
    info1 = mne.create_info(ch_names=['CH1', 'CH2', 'CH3', 'CH4', 'CH5', 'CH6', 'CH7', 'CH8'], sfreq=1000)
    raw1 = mne.io.RawArray(data1_raw, info1)
    raw1.notch_filter(np.arange(50, 451, 50), picks='all')
    raw1.filter(4, 50, method='iir', picks='all')
    data1_filt = raw1.get_data()
    epoch1_filt.append(segment_epoch(data1_filt.copy(), event1, tmin, tmax, fs))

# Stack
epoch1_raw = np.concatenate(epoch1_raw, 0)
epoch1_filt = np.concatenate(epoch1_filt, 0)
epoch1_filt = epoch1_filt[:, :, int(offset*fs):]
label1 = np.concatenate(label1, 0)
n_block1 = int(len(label1) / n_target)


### CNT
epoch2_raw, epoch2_filt, label2 = [], [], []
for idx_cnt in range(1, 4):
    # Load
    raw2 = mne.io.read_raw_cnt('./demos/brainflow_demos/Data/yyy_offline{}.cnt'.format(idx_cnt), preload=True)
    data2_raw = raw2.get_data()
    data2_raw = data2_raw[56:64].copy()
    
    # Event
    event2, _ = mne.events_from_annotations(raw2, event_id=event_map)
    label2.append(event2[:, -1]-1)
    
    # Raw for classification
    epoch2_raw.append(segment_epoch(data2_raw.copy(), event2, tmin, tmax, fs))
    
    # Filtering for analysis
    raw2.filter(4, 50, method='iir', picks='all')
    data2_filt = raw2.get_data()
    data2_filt = data2_filt[56:64].copy()
    epoch2_filt.append(segment_epoch(data2_filt.copy(), event2, tmin, tmax, fs))

# Stack
epoch2_raw = np.concatenate(epoch2_raw, 0)
epoch2_filt = np.concatenate(epoch2_filt, 0)
epoch2_filt = epoch2_filt[:, :, int(offset*fs):]
label2 = np.concatenate(label2, 0)
n_block2 = int(len(label2) / n_target)


# ### BDF
epoch3_raw, epoch3_filt, label3 = [], [], []
for idx_bdf in range(1, 4):
    # Load
    raw3 = mne.io.read_raw_bdf('./demos/brainflow_demos/Data/yyy_offline{}.bdf'.format(idx_bdf), preload=True)
    
    # Interpolation
    data3 = raw3.get_data()
    for idx_chan in range(n_chan):
        m, s = data3[idx_chan].mean(), data3[idx_chan].std()
        idx_bad = np.where((data3[idx_chan]>m+3*s) | (data3[idx_chan]<m-3*s))[0]
        # print(len(idx_bad))
        for i_bad in idx_bad:
            i_left = i_bad-25 if i_bad >= 25 else 0
            i_right = i_bad+25 if i_bad <=len(data3[idx_chan])-25 else len(data3[idx_chan])
            data3[idx_chan, i_bad] = np.median(data3[idx_chan, i_left:i_right])
    
    # Event
    event3, _ = mne.events_from_annotations(raw3, event_id=event_map)
    label3.append(event3[:, -1]-1)
    
    # Raw for classification
    data3_raw = data3.copy()
    epoch3_raw.append(segment_epoch(data3_raw.copy(), event3, tmin, tmax, fs))
    
    # Filtering for analysis
    info3 = raw3.info
    raw3 = mne.io.RawArray(data3, info3)
    raw3.notch_filter(np.arange(50, 451, 50), picks='all')
    raw3.filter(4, 50, method='iir', picks='all')
    data3_filt = raw3.get_data()
    epoch3_filt.append(segment_epoch(data3_filt.copy(), event3, tmin, tmax, fs))

# Stack
epoch3_raw = np.concatenate(epoch3_raw, 0)
epoch3_filt = np.concatenate(epoch3_filt, 0)
epoch3_filt = epoch3_filt[:, :, int(offset*fs):]
label3 = np.concatenate(label3, 0)
n_block3 = int(len(label3) / n_target)


# In[]: Time domain
evoke1 = np.stack([epoch1_filt[label1==k].mean((0, 1)) for k in range(n_target)], 0)
evoke2 = np.stack([epoch2_filt[label2==k].mean((0, 1)) for k in range(n_target)], 0)
evoke3 = np.stack([epoch3_filt[label3==k].mean((0, 1)) for k in range(n_target)], 0)

font_title = FontProperties(family='Arial', size=18, weight='bold')
font_label = FontProperties(family='Arial', size=10, weight='bold')
font_tick = FontProperties(family='Arial', size=10, weight='bold')
t_show = 1
x = np.linspace(0, t_show, fs)
fig = plt.figure(figsize=(8, 9), tight_layout=True)
for idx_e, evoke in enumerate([evoke1, evoke2, evoke3]):
    for idx_tar in range(n_target):
        # Normalization
        y = (evoke[idx_tar, :int(t_show*fs)]-evoke[idx_tar, :int(t_show*fs)].mean())/evoke[idx_tar, :int(t_show*fs)].std()
        
        # Plot
        plt.subplot(n_target, 3, idx_tar*3+idx_e+1)
        plt.plot(x, y, color='darkblue', linewidth=1)
        plt.ylim([-3, 3])
        plt.xlim([0, t_show])
        # plt.xticks(np.arange(0, t_show+0.1, 0.2), np.round(np.arange(0, t_show+0.1, 0.2), 1))
        plt.yticks([0, 3], [0, 3], fontproperties=font_tick)
        # plt.xticks(np.arange(0, t_show+0.1, 0.2), [])
        plt.xticks(np.arange(0, t_show+0.1, 0.2), np.round(np.arange(0, t_show+0.1, 0.2), 1), fontproperties=font_tick)
        plt.gca().tick_params(axis='x', direction='out')
        
        if idx_e == 0:
            if idx_tar == 0:
                plt.title('CSV(Time)', fontproperties=font_title)
            # plt.ylabel('{} Hz'.format(freqs[idx_tar]), rotation=0, ha='right', va='center', fontproperties=font_title)
            plt.ylabel('{} Hz\nAmp.(Norm)'.format(freqs[idx_tar]), fontproperties=font_label)
            # plt.yticks([-3, 0, 3], [-3, 0, 3], fontproperties=font_tick)
        if idx_e == 1 and idx_tar == 0:
            plt.title('CNT(Time)', fontproperties=font_title)
        if idx_e == 2 and idx_tar == 0:
            plt.title('BDF(Time)', fontproperties=font_title)
        if idx_tar == n_target-1:
            plt.xlabel('Time(s)', fontproperties=font_label)
            plt.xticks(np.arange(0, t_show+0.1, 0.2), np.round(np.arange(0, t_show+0.1, 0.2), 1), fontproperties=font_tick)

# plt.savefig('time.png', dpi=300)


# In[]: Spectrum domain
evoke1_spec = np.zeros((evoke1.shape[0], round(evoke1.shape[1]/2)))
for idx_tar in range(n_target):
    y1 = (evoke1[idx_tar]-evoke1[idx_tar].mean())/evoke1[idx_tar].std()
    evoke1_spec[idx_tar], _, f = get_spectrum(y1, fs)

evoke2_spec = np.zeros((evoke2.shape[0], round(evoke2.shape[1]/2)))
for idx_tar in range(n_target):
    y2 = (evoke2[idx_tar]-evoke2[idx_tar].mean())/evoke2[idx_tar].std()
    evoke2_spec[idx_tar], _, f = get_spectrum(y2, fs)
    
evoke3_spec = np.zeros((evoke3.shape[0], round(evoke3.shape[1]/2)))
for idx_tar in range(n_target):
    y3 = (evoke3[idx_tar]-evoke3[idx_tar].mean())/evoke3[idx_tar].std()
    evoke3_spec[idx_tar], _, f = get_spectrum(y3, fs)
    
    
font_title = FontProperties(family='Arial', size=18, weight='bold')
font_label = FontProperties(family='Arial', size=10, weight='bold')
font_tick = FontProperties(family='Arial', size=10, weight='bold')
fig = plt.figure(figsize=(8, 9), tight_layout=True)
for idx_e, evoke_spec in enumerate([evoke1_spec, evoke2_spec, evoke3_spec]):
    for idx_tar in range(n_target):
        y = evoke_spec[idx_tar]
        
        # Plot
        plt.subplot(n_target, 3, idx_tar*3+idx_e+1)
        plt.plot(f, y, color='darkblue', linewidth=1)
        plt.plot(freqs[idx_tar], y[[np.where(f==freqs[idx_tar])[0]]],
                    marker='o', c='r', linestyle='', markerfacecolor='none')
        plt.plot(freqs[idx_tar]*2, y[[np.where(f==freqs[idx_tar]*2)[0]]],
                    marker='o', c='r', linestyle='', markerfacecolor='none')
        plt.ylim([0, 1])
        plt.xlim([0, 50])
        plt.yticks([0.5, 1], [0.5, 1], fontproperties=font_tick)
        plt.xticks(np.arange(0, 51, 10), np.round(np.arange(0, 51, 10)), fontproperties=font_tick)
        
        plt.gca().tick_params(axis='x', direction='out')
        
        if idx_e == 0:
            if idx_tar == 0:
                plt.title('CSV(Spec)', fontproperties=font_title)
            plt.ylabel('{} Hz\nAmp.(Norm)'.format(freqs[idx_tar]), fontproperties=font_label)
            # plt.yticks([-3, 0, 3], [-3, 0, 3], fontproperties=font_tick)
        if idx_e == 1 and idx_tar == 0:
            plt.title('CNT(Spec)', fontproperties=font_title)
        if idx_e == 2 and idx_tar == 0:
            plt.title('BDF(Spec)', fontproperties=font_title)
        if idx_tar == n_target-1:
            plt.xlabel('Frequency(Hz)', fontproperties=font_label)
            plt.xticks(np.arange(0, 51, 10), np.round(np.arange(0, 51, 10)), fontproperties=font_tick)

# plt.savefig('spce.png', dpi=300)


# In[]: Classification
# Notch filter
f0 = 50
Q = 30
b_notch, a_notch = signal.iirnotch(f0, Q, fs=fs)

# TRCA decoding model
model = TRCA()
# model.init_filter(passband=[[8, 90], [16, 90], [24, 90], [32, 90], [40, 90]], 
#                   stopband=[[6, 100], [14, 100], [22, 100], [30, 100], [38, 100]],
#                   gpass=3,
#                   gstop=10,
#                   rp=0.5,
#                   fs=fs)

# Run for different time length
t_list = np.round(np.arange(0.5, 2.1, 0.1), 1)
acc1, acc2, acc3 = np.zeros((len(t_list))), np.zeros((len(t_list))), np.zeros((len(t_list)))
for idx_t, t in enumerate(t_list):
    # Notch filtering for suit time length
    epoch1 = signal.filtfilt(b_notch, a_notch, epoch1_raw[..., :int(offset*fs+t*fs)])
    epoch2 = epoch2_raw[..., :int(offset*fs+t*fs)]  # CNT have been notch-filtered
    epoch3 = signal.filtfilt(b_notch, a_notch, epoch3_raw[..., :int(offset*fs+t*fs)])
    epoch1 = epoch1[..., int(offset*fs):]
    epoch2 = epoch2[..., int(offset*fs):]
    epoch3 = epoch3[..., int(offset*fs):]

    # Leave-one-block-out cross validation
    result1 = leave_one_block_out_test(epoch1, label1, model, n_block=n_block1)
    result2 = leave_one_block_out_test(epoch2, label2, model, n_block=n_block2)
    result3 = leave_one_block_out_test(epoch3, label3, model, n_block=n_block3)
    
    # Accuracy
    acc1[idx_t] = np.mean(result1)
    acc2[idx_t] = np.mean(result2)
    acc3[idx_t] = np.mean(result3)
    print('Time:{} CSV:{:.3f} CNT:{:.3f} BDF:{:.3f}'.format(t, acc1[idx_t], acc2[idx_t], acc3[idx_t]))
    
    
# In[]:
font_title = FontProperties(family='Arial', size=20, weight='bold')
font_label = FontProperties(family='Arial', size=18, weight='bold')
font_tick = FontProperties(family='Arial', size=14, weight='bold')
fig = plt.figure(figsize=(6, 6), tight_layout=True)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
    
plt.plot(t_list, acc1, color='b', marker='o', markerfacecolor='white', linewidth=1.5, markeredgewidth=1.5, clip_on=False,
          label='CSV ({}-cross)'.format(n_block1))
plt.plot(t_list, acc2, color='r', marker='d', markerfacecolor='white', linewidth=1.5, markeredgewidth=1.5, clip_on=False,
          label='CNT ({}-cross)'.format(n_block2))
plt.plot(t_list, acc3, color='k', marker='s', markerfacecolor='white', linewidth=1.5, markeredgewidth=1.5, clip_on=False,
          label='BDF ({}-cross)'.format(n_block3))
plt.xlim(0.45, 2.05)
plt.ylim(0, 1)
plt.xticks(np.round(np.arange(0.5, 2.1, 0.3), 1), fontproperties=font_tick)
plt.yticks(np.round(np.arange(0, 1.1, 0.2), 1), fontproperties=font_tick)
plt.xlabel('Time (s)', fontproperties=font_label)
plt.ylabel('Accuracy', fontproperties=font_label)
plt.title('Leave-one-block-out Cross Validation', fontproperties=font_title)
plt.legend(loc='lower right', frameon=False)
plt.grid(axis='y')
# plt.savefig('LOOCV.png', dpi=300)
