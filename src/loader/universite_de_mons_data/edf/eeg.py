import numpy as np
import pyedflib

from src.utils.util import *
from src.logger.messages import *
import src.logger.logger as logger


class EEG:
    def __init__(self, path):
        self.__path = path
        self.__edf = self.__load_edf_data(path)
        self.__eeg_signals = self.__load_all_eeg()

    def __load_edf_data(self, path):
        logger.info(EEG_LOAD_START_INFO)
        edf = None
        try:
            edf = pyedflib.EdfReader(path)
            logger.info(EEG_LOAD_END_INFO)
        except OSError:
            logger.error(EEG_LOAD_ERROR)

        return edf

    def __load_all_eeg(self):
        logger.info(EEG_SIGNAL_SELECTION_START_INFO)
        sample_nos = [LOAD_EDF_EEG_FPI_A2_NO, LOAD_EDF_EEG_CZ_A1_NO, LOAD_EDF_EEG_O1_A2_NO,
                      LOAD_EDF_EEG_FP2_A1_NO, LOAD_EDF_EEG_O2_A1_NO, LOAD_EDF_EEG_CZ2_A1_NO]
        if self.__edf:
            eeg_samples = np.zeros((len(sample_nos), self.__edf.getNSamples()[EEG_FPI_A2_NO]))
            for i in range(len(sample_nos)):
                sample_no = sample_nos[i]
                eeg_samples[i, :] = self.__edf.readSignal(sample_no)
            logger.info(EEG_SIGNAL_SELECTION_END_INFO)
        else:
            eeg_samples = [[] for _ in sample_nos]
            logger.warning(EEG_SIGNAL_SELECTION_END_WARNING)
        return eeg_samples

    def load_fpi_a2(self):
        return self.__eeg_signals[EEG_FPI_A2_NO]

    def load_cz_a1(self):
        return self.__eeg_signals[EEG_CZ_A1_NO]

    def load_o1_a2(self):
        return self.__eeg_signals[EEG_O1_A2_NO]

    def load_fp2_a1(self):
        return self.__eeg_signals[EEG_FP2_A1_NO]

    def load_o2_a1(self):
        return self.__eeg_signals[EEG_O2_A1_NO]

    def load_cz2_a1(self):
        return self.__eeg_signals[EEG_CZ2_A1_NO]

    # Wszystkie sygnały EEG mają tą samą częstotliwość - 200 Hz
    def load_eeg_frequency(self):
        freq = 0
        if self.__edf:
            freq = self.__edf.getSampleFrequency(EEG_FPI_A2_NO)
        return freq
