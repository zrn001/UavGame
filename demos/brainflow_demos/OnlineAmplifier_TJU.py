# -*- coding: utf-8 -*-
"""
SSAVEP Feedback on NeuroScan.

"""
import time
import numpy as np

import mne
from mne.filter import resample
from pylsl import StreamInfo, StreamOutlet
from scipy import signal, fftpack
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
from metabci.brainflow.amplifiers import NeuroScan, New_Uestc, Marker
from metabci.brainflow.workers import ProcessWorker
from metabci.brainda.algorithms.decomposition.base import (
    generate_filterbank, generate_cca_references)
from metabci.brainda.algorithms.utils.model_selection import (
    EnhancedLeaveOneGroupOut)
from metabci.brainda.algorithms.decomposition import FBTDCA,FBTRCA
from metabci.brainda.utils import upper_ch_names
from mne.io import read_raw_cnt
from sklearn.base import BaseEstimator, ClassifierMixin


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


def read_data(file_path, chs, interval, labels, srate=2000):
    ##########################################################################################
    # In[2]: Load data
    epoch_raw, epoch_filt, label = [], [], []
    for idx_bdf in range(len(file_path)):
        # Load data
        raw = mne.io.read_raw_bdf(file_path[idx_bdf], preload=True)
        # Interpolation for bad points
        data = raw.get_data()
        event, _ = mne.events_from_annotations(raw, event_id=labels)
        label.append(event[:, -1]-1)

        # Raw for classification
        data_raw = data.copy()
        epoch_raw.append(segment_epoch(data_raw.copy(), event, interval[0], interval[1], srate))
        
        # Filtering for analysis
        info = raw.info
        raw = mne.io.RawArray(data, info)
        raw.notch_filter(50, picks='all')
        raw.filter(4, 50, method='iir', picks='all')
        data_filt = raw.get_data()
        epoch_filt.append(segment_epoch(data_filt.copy(), event, interval[0], interval[1], srate))

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
        evoke_spec[idx_tar], _, f = get_spectrum(evoke[idx_tar], srate)

    # FFT to Spectrum domain
    evoke_spec = np.zeros((evoke.shape[0], round(evoke.shape[1]/2)))
    for idx_tar in range(n_target):
        evoke_spec[idx_tar], _, f = get_spectrum(evoke[idx_tar], srate)
        

    # In[3]: Time domain & Spectrum domain
    font_title = FontProperties(size=18, weight='bold')
    font_label = FontProperties(size=9, weight='bold')
    font_tick = FontProperties(size=9, weight='bold')
    t_show = 1
    x = np.linspace(0, t_show, srate)
    fig = plt.figure(figsize=(6, 9), tight_layout=True)

    for idx_tar in range(n_target):
        # Standardization
        # y = (evoke[idx_tar, :int(t_show*fs)]-evoke[idx_tar, :int(t_show*fs)].mean())/evoke[idx_tar, :int(t_show*fs)].std()
        y = evoke[idx_tar, :int(t_show*srate)]
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
    ##########################################################################################

def train_model(X, y, label, srate=1000):
    # In[4]: Classification
    # Notch filter
    f0 = 50
    Q = 30
    b_notch, a_notch = signal.iirnotch(f0, Q, fs=srate)
    n_block3 = int(len(label) / n_target)

    # TRCA decoding model
    # wp = [(8,48), (16,48), (24,48)]
    # ws = [(6,50), (14,50), (22,50)]
    wp = [(5,90), (14,90), (22,90), (30,90), (38,90)]
    ws = [(3,92), (12,92), (20,92), (28,92), (36,92)]
    filterweights = np.array([(idx_filter+1)**(-1.25)+0.25 for idx_filter in range(len(wp))])
    filterbank = generate_filterbank(wp, ws, order=15, srate=srate, rp=0.5)
    model = FBTRCA(filterbank=filterbank, n_components=1, ensemble=True, filterweights=filterweights, n_jobs=-1)

    # Run for different time length
    t_list = np.round(np.arange(0.5, 2.1, 0.1), 1)
    acc = np.zeros((len(t_list)))
    for idx_t, t in enumerate(t_list):
        t0 = time.time()
        # Notch filtering for suit time length
        epoch = signal.filtfilt(b_notch, a_notch, epoch_raw[..., :int(t*srate)]) # ---------------有问题--------------
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

    return model


def model_predict(X, srate=1000, model=None):
    X = np.reshape(X, (-1, X.shape[-2], X.shape[-1]))
    X = resample(X, up=256, down=srate)
    X = X - np.mean(X, axis=-1, keepdims=True)
    X = X / np.std(X, axis=(-1, -2), keepdims=True)
    p_labels = model.predict(X)
    return p_labels


def offline_validation(X, y,label, srate=1000):
    y = np.reshape(y, (-1))
    spliter = EnhancedLeaveOneGroupOut(return_validate=False)

    kfold_accs = []
    for train_ind, test_ind in spliter.split(X, y=y):
        X_train, y_train = np.copy(X[train_ind]), np.copy(y[train_ind])
        X_test, y_test = np.copy(X[test_ind]), np.copy(y[test_ind])

        model = train_model(X_train, y_train,label, srate=srate)
        p_labels = model_predict(X_test, srate=srate, model=model)
        kfold_accs.append(np.mean(p_labels == y_test))
    return np.mean(kfold_accs)


class FeedbackWorker(ProcessWorker):
    def __init__(self, run_files, pick_chs, stim_interval, stim_labels,
                 srate, lsl_source_id, timeout, worker_name):
        self.run_files = run_files
        self.pick_chs = pick_chs
        self.stim_interval = stim_interval
        self.stim_labels = stim_labels
        self.srate = srate
        self.lsl_source_id = lsl_source_id
        super().__init__(timeout=timeout, name=worker_name)

    def pre(self):
        X, y, ch_ind = read_data(run_files=self.run_files,
                                  chs=self.pick_chs,
                                  interval=self.stim_interval,
                                  labels=self.stim_labels,
                                  srate=self.srate)
        
        print("Loding train data successfully")
        # Compute offline acc
        acc = offline_validation(X, y,stim_labels, srate=self.srate)
        print("Current Model accuracy:{:.2f}".format(acc))
        self.estimator = train_model(X, y, srate=self.srate)
        # self.ch_ind = ch_ind

        info = StreamInfo(
            name='meta_feedback',
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
        data = data[self.ch_ind]
        p_labels = model_predict(data, srate=self.srate, model=self.estimator)
        p_labels = np.array([int(p_labels + 1)])
        # p_labels = p_labels.tolist()
        p_labels = list(p_labels)
        print('predict_id_paradigm', p_labels)
        if self.outlet.have_consumers():
            self.outlet.push_sample(p_labels)

    def post(self):
        pass


if __name__ == '__main__':
    ##########################################################################################
    # Sample rate EEG amplifier

    # In[1]: Config
    srate = 250
    stim_labels = {str(e):e for e in range(1, 9)}
    # stim_labels = list(range(1, 9))
    n_chan = 8
    freqs = [8, 9, 10, 11, 12, 13, 14, 15]
    n_target = len(freqs)
    run_files = ['./Data/Tju-testOffline/1128/1.bdf', 
                 './Data/Tju-testOffline/1128/2.bdf']
    stim_interval = [0.14, 2.14]
    pick_chs = ['Channel_1', 'Channel_2', 'Channel_3', 'Channel_4', 'Channel_5', 'Channel_6', 'Channel_7', 'Channel_8', ]
    ##########################################################################################

    lsl_source_id = 'mobile-china'
    feedback_worker_name = 'feedback_worker'

    worker = FeedbackWorker(
        run_files=run_files,
        pick_chs=pick_chs,
        stim_interval=stim_interval,
        stim_labels=stim_labels, srate=srate,
        lsl_source_id=lsl_source_id,
        timeout=5e-2,
        worker_name=feedback_worker_name)
    marker = Marker(interval=stim_interval, srate=srate,
                    events=stim_labels)

    worker.pre()
    # Set New_Uestc parameters
    # ns = New_Uestc(
    #     device_address=('127.0.0.1', 9687),
    #     srate=srate,
    #     num_chans=8)

    # Start tcp connection with ns
    # ns.connect_tcp()
    # # Start acquire data from ns
    # ns.start_acq()

    # # Register worker for online data processing56
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
