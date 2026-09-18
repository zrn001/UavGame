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



class NanoEEG(BaseAmplifier):

    def __init__(self,
                 device_address: Tuple[str, int] = ('127.0.0.1', 1895),
                 srate=1000,
                 num_chans=16):
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
            data = self.tcp_link.recv(9216)
           # print(data)
        except Exception as e:
            print("连接出错", e)
            self.tcp_link.close()
        finally:
            #
            if data is not None:
                self.data_count +=1
                data = self.__upack_data(data)
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

    def segment_epoch(self):
        """ Segment epochs. """
        # Segment epochs
        pass
        

    def read_data(self):
        """ Read data. """
        pass
        

    def pre(self):
        # print("pre:cca")
        # Read data
        p_labels_cca = self.model.predict(data)
        print('ACC: {}'.format(np.mean(p_labels_cca == labels)))

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
        # print("consume:cca")
        p_labels_cca = self.model.predict(data)

        k = 0
        my_list = [[1], [2], [3], [4], [5], [6], [7], [8]]
        cre_list.append(p_labels_cca+1)
        for i in range(len(cre_list)):
            if cre_list[i] == my_list[i % 8]:
                k = k + 1
        print("在线识别准确率：", float(k)/float(len(cre_list)))


    def post(self):
        pass


if __name__ == '__main__':
    run_files = ['data/ssvep_20230424_cca_1.mat']
    pick_chs = ['O1', 'O2', 'Oz', 'PO3', 'PO4', 'POz']
    stim_interval = [0.0, 4.0]  

    lsl_source_id = 'mobile-bci'
    feedback_worker_name = 'nano_worker'

    worker = FeedbackWorker(
        run_files=run_files,
        pick_chs=pick_chs,
        stim_interval=stim_interval,
        event_map=event_map, srate=srate,
        lsl_source_id=lsl_source_id,
        timeout=5e-2,
        worker_name=feedback_worker_name)
    marker = Marker(stim_interval,srate,stim_labels)

    worker.pre()
    worker.consume(marker.get_epoch())
    # # Set Neuroscan parameters
    
    ns = NanoEEG()
    ns.register_worker(feedback_worker_name, worker, marker)
    ns.up_worker(feedback_worker_name)
    time.sleep(0.5)
    ns.start_trans()

    # Start tcp connection with ns
    ns.connect_tcp()
    # Start acquire data from ns
    ns.start_acq()

    # Register worker for online data processing
    ns.register_worker(feedback_worker_name, worker, marker)
    # Start online data processing
    ns.up_worker(feedback_worker_name)
    time.sleep(0.5)

    # Start slicing data and passing data to worker
    ns.start_trans()

    input('press any key to close\n')
    ns.down_worker('feedback_worker')
    time.sleep(1)

    # Stop online data retriving of ns
    ns.stop_trans()
    ns.stop_acq()
    ns.close_connection()
    ns.clear()
    print('bye')
