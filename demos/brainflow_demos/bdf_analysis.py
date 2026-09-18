import time
import mne
import numpy as np
from scipy import signal, fftpack
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from metabci.brainda.algorithms.decomposition import FBTRCA
from metabci.brainda.algorithms.decomposition.base import generate_filterbank
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


# In[1]: Config
event_map = {str(e):e for e in range(1, 9)}
fs = 250
tmax = 5.14
tmin = 0.14
n_chan = 8
freqs = [8, 9, 10, 11, 12, 13, 14, 15]
n_target = len(freqs)
file_path = ['./Data/Tju-testOffline/1201/offline_text3.bdf']


# In[2]: Load data
epoch_raw, epoch_filt, label = [], [], []
for idx_bdf in range(len(file_path)):
    # Load data
    raw = mne.io.read_raw_bdf(file_path[idx_bdf], preload=True)
    # Interpolation for bad points
    data = raw.get_data()
    # for idx_chan in range(n_chan):
    #     m, s = data[idx_chan].mean(), data[idx_chan].std()
    #     idx_bad = np.where((data[idx_chan]>m+3*s) | (data[idx_chan]<m-3*s))[0]
    #     for i_bad in idx_bad:
    #         i_left = i_bad-50 if i_bad >= 50 else 0
    #         data[idx_chan, i_bad] = np.median(data[idx_chan, i_left:i_bad])        
    
    # Event
    event, _ = mne.events_from_annotations(raw, event_id=event_map)
    label.append(event[:, -1]-1)
    
    # Raw for classification
    data_raw = data.copy()
    epoch_raw.append(segment_epoch(data_raw.copy(), event, tmin, tmax, fs))
    
    # Filtering for analysis
    info = raw.info
    raw = mne.io.RawArray(data, info)
    raw.notch_filter(50, picks='all')
    raw.filter(4, 50, method='iir', picks='all')
    data_filt = raw.get_data()
    epoch_filt.append(segment_epoch(data_filt.copy(), event, tmin, tmax, fs))

# Stack
epoch_raw = np.concatenate(epoch_raw, 0)
epoch_filt = np.concatenate(epoch_filt, 0)
label = np.concatenate(label, 0)
n_block3 = int(len(label) / n_target)

# Mean across trials and channels
evoke = np.stack([epoch_filt[label==k].mean((0, 1)) for k in range(n_target)], 0)
# Standardization
evoke = (evoke-evoke.mean(-1, keepdims=True))/evoke.std(-1, keepdims=True)

# FFT to Spectrum domain
evoke_spec = np.zeros((evoke.shape[0], round(evoke.shape[1]/2)))
for idx_tar in range(n_target):
    evoke_spec[idx_tar], _, f = get_spectrum(evoke[idx_tar], fs)
    

# In[3]: Time domain & Spectrum domain
font_title = FontProperties(size=18, weight='bold')
font_label = FontProperties(size=9, weight='bold')
font_tick = FontProperties(size=9, weight='bold')
t_show = 1
x = np.linspace(0, t_show, fs)
fig = plt.figure(figsize=(6, 9), tight_layout=True)

for idx_tar in range(n_target):
    # Standardization
    # y = (evoke[idx_tar, :int(t_show*fs)]-evoke[idx_tar, :int(t_show*fs)].mean())/evoke[idx_tar, :int(t_show*fs)].std()
    y = evoke[idx_tar, :int(t_show*fs)]
    y_spec = evoke_spec[idx_tar]
    
    # Plot data (Time)
    plt.subplot(n_target, 2, idx_tar*2 + 1)
    plt.plot(x, y, color='darkblue', linewidth=1)   
    # Plot config
    plt.xlim([0, t_show])
    plt.ylim([-3, 3])
    plt.xticks(np.arange(0, t_show+0.1, 0.2), np.round(np.arange(0, t_show+0.1, 0.2), 1), fontproperties=font_tick)
    plt.yticks([0, 3], [0, 3], fontproperties=font_tick)
    plt.gca().tick_params(axis='x', direction='out')
    plt.ylabel('{} Hz\nAmp.(Norm)'.format(freqs[idx_tar]), fontproperties=font_label)
    if idx_tar == 0:
        plt.title('Time Domain', fontproperties=font_title)
    if idx_tar == n_target-1:
        plt.xlabel('Time(s)', fontproperties=font_label)

    # Plot data (Spectrum)
    plt.subplot(n_target, 2, idx_tar*2 + 2)
    plt.plot(f, y_spec, color='darkblue', linewidth=1)
    plt.plot([freqs[idx_tar], freqs[idx_tar]*2, freqs[idx_tar]*3], y_spec[(f==freqs[idx_tar])+(f==2*freqs[idx_tar])+(f==3*freqs[idx_tar])],
                marker='o', c='r', linestyle='', markerfacecolor='none')
    plt.xlim([0, 50])
    plt.ylim([0, 1])
    plt.xticks(np.arange(0, 51, 10), np.round(np.arange(0, 51, 10)), fontproperties=font_tick)
    plt.yticks([0.5, 1], [0.5, 1], fontproperties=font_tick)
    plt.gca().tick_params(axis='x', direction='out')
    # plt.ylabel('{} Hz\nAmp.(Norm)'.format(freqs[idx_tar]), fontproperties=font_label)
    if idx_tar == 0:
        plt.title('Spectrum Domain', fontproperties=font_title)
    if idx_tar == n_target-1:
        plt.xlabel('Frequency(Hz)', fontproperties=font_label)

plt.savefig('Feature.png', dpi=300)
plt.show()


# In[4]: Classification
# Notch filter
f0 = 50
Q = 30
b_notch, a_notch = signal.iirnotch(f0, Q, fs=fs)

# TRCA decoding model
# wp = [(8,48), (16,48), (24,48)]
# ws = [(6,50), (14,50), (22,50)]
wp = [(5,90), (14,90), (22,90), (30,90), (38,90)]
ws = [(3,92), (12,92), (20,92), (28,92), (36,92)]
filterweights = np.array([(idx_filter+1)**(-1.25)+0.25 for idx_filter in range(len(wp))])
filterbank = generate_filterbank(wp, ws, order=15, srate=fs, rp=0.5)
model = FBTRCA(filterbank=filterbank, n_components=1, ensemble=True, filterweights=filterweights, n_jobs=-1)

# Run for different time length
t_list = np.round(np.arange(0.5, 5.1, 0.1), 1)
acc = np.zeros((len(t_list)))
for idx_t, t in enumerate(t_list):
    t0 = time.time()
    # Notch filtering for suit time length
    epoch = signal.filtfilt(b_notch, a_notch, epoch_raw[..., :int(t*fs)])
    # Leave-one-block-out cross validation
    result = leave_one_block_out_test(epoch, label, model, n_block=n_block3)
    # Accuracy
    acc[idx_t] = np.mean(result)
    print('Time:{} Acc:{:.3f}'.format(t, acc[idx_t]))

    model.fit(epoch, label)
    print('Training time:{:.4f}'.format(time.time()-t0))

# Plot accuracy
font_title = FontProperties(family='Arial', size=20, weight='bold')
font_label = FontProperties(family='Arial', size=18, weight='bold')
font_tick = FontProperties(family='Arial', size=14, weight='bold')
fig = plt.figure(figsize=(6, 6), tight_layout=True)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.plot(t_list, acc, color='k', marker='s', markerfacecolor='white', linewidth=1.5, markeredgewidth=1.5, clip_on=False,
          label='BDF ({}-fold)'.format(n_block3))
plt.xlim(0.45, 2.05)
plt.ylim(0, 1)
plt.xticks(np.round(np.arange(0.5, 2.1, 0.3), 1), fontproperties=font_tick)
plt.yticks(np.round(np.arange(0, 1.1, 0.2), 1), fontproperties=font_tick)
plt.xlabel('Time (s)', fontproperties=font_label)
plt.ylabel('Accuracy', fontproperties=font_label)
plt.title('Leave-one-block-out Cross Validation', fontproperties=font_title)
plt.legend(loc='lower right', frameon=False)
plt.grid(axis='y')
plt.savefig('Accuracy.png', dpi=300)