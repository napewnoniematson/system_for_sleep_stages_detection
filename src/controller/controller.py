import matplotlib.pyplot as plt
import random

from src.model.universite_de_mons_data.examination import Examination, HypnogramType
from src.feature.provider import Provider
import src.filter.butterworth as bf
from src.utils.util import *
from src.ai.deep_learning import DeepLearner

class Controller:
    SUBJECT_NO = 16

    def __init__(self):
        self.examination = Examination(self.SUBJECT_NO)
        self.fpi_a2 = self.examination.eeg.load_fpi_a2()

    def run(self):
        # plt.plot(self.examination.hypnogram["data"])
        # plt.show()
        fs = self.examination.eeg.load_eeg_frequency()
        low_cut = 0.5
        high_cut = 30
        filtered = bf.butter_bandpass_filter(self.fpi_a2, low_cut, high_cut, fs, order=6)

        wake = self.examination.hypnogram['wake']
        rem = self.examination.hypnogram['rem']
        s1 = self.examination.hypnogram['s1']
        s2 = self.examination.hypnogram['s2']
        s3 = self.examination.hypnogram['s3']

        length = len(wake) + len(rem) + len(s1) + len(s2) + len(s3)
        fp = Provider(filtered, length)
        fp.update(wake, 5)
        fp.update(rem, 4)
        fp.update(s1, 3)
        fp.update(s2, 2)
        fp.update(s3, 1)

        data_set = fp.get_data_set()
        random.shuffle(data_set)
        random.shuffle(data_set)
        random.shuffle(data_set)
        split_index = int(length * 0.8)
        training = data_set[:split_index]
        test = data_set[split_index:]

        print(data_set)

        dl = DeepLearner()
