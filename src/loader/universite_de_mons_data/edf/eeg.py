import numpy as np
import pyedflib

from src.utils.util import *


class EEG:
    def __init__(self, path):
        self.__path = path
        self.__edf = pyedflib.EdfReader(path)
        self.__eeg_signals = self.__load_all_eeg()

    def __load_all_eeg(self):
        sample_nos = [EEG_FPI_A2_NO, EEG_CZ_A1_NO, EEG_O1_A2_NO,
                      EEG_FP2_A1_NO, EEG_O2_A1_NO, EEG_CZ2_A1_NO]
        eeg_samples = np.zeros((len(sample_nos), self.__edf.getNSamples()[EEG_FPI_A2_NO]))
        for i in range(len(sample_nos)):
            sample_no = sample_nos[i]
            eeg_samples[i, :] = self.__edf.readSignal(sample_no)
        return eeg_samples

    def load_fpi_a2(self):
        return self.__eeg_signals[EEG_FPI_A2_NO]

    def load_cz_a1(self):
        return self.__eeg_signals[EEG_CZ_A1_NO]

    def load_o1_a2(self):
        return self.__edf.__eeg_signals[EEG_O1_A2_NO]

    def load_fp2_a1(self):
        return self.__edf.__eeg_signals[EEG_FP2_A1_NO]

    def load_o2_a1(self):
        return self.__edf.__eeg_signals[EEG_O2_A1_NO]

    def load_cz2_a1(self):
        return self.__edf.__eeg_signals[EEG_CZ2_A1_NO]
