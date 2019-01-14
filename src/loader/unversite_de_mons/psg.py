import numpy as np
import pyedflib

from src.utils.util import *
from src.logger.messages import *
import src.logger.logger as logger


# todo change logs
class PSG:
    def __init__(self, path):
        self.__path = path
        self.__edf = self.__load_edf_data(path)
        self.__signals = self.__load_signals()

    @staticmethod
    def __load_edf_data(path):
        logger.info(EEG_LOAD_START_INFO)
        edf = None
        try:
            edf = pyedflib.EdfReader(path)
            logger.info(EEG_LOAD_END_INFO)
        except OSError:
            logger.error(EEG_LOAD_ERROR)
        return edf

    def __load_signals(self):
        logger.info(EEG_SIGNAL_SELECTION_START_INFO)
        channels_no = self.__get_channels_no()
        if self.__edf:
            samples = np.zeros((len(channels_no), self.__load_samples_amount()))
            for i in range(len(channels_no)):
                sample_no = channels_no[i]
                samples[i, :] = self.__edf.readSignal(sample_no)
            logger.info(EEG_SIGNAL_SELECTION_END_INFO)
        else:
            samples = [[] for _ in channels_no]
            logger.warning(EEG_SIGNAL_SELECTION_END_WARNING)
        return samples

    @staticmethod
    def __get_channels_no():
        return [LOAD_EDF_EEG_FPI_A2_NO, LOAD_EDF_EEG_CZ_A1_NO, LOAD_EDF_EEG_O1_A2_NO,
                LOAD_EDF_EEG_FP2_A1_NO, LOAD_EDF_EEG_O2_A1_NO, LOAD_EDF_EEG_CZ2_A1_NO,
                LOAD_EDF_EOG1_A2_NO, LOAD_EDF_EOG2_A2_NO]

    def __load_fpi_a2(self):
        return self.__signals[EEG_FPI_A2_NO]

    def __load_cz_a1(self):
        return self.__signals[EEG_CZ_A1_NO]

    def __load_o1_a2(self):
        return self.__signals[EEG_O1_A2_NO]

    def __load_fp2_a1(self):
        return self.__signals[EEG_FP2_A1_NO]

    def __load_o2_a1(self):
        return self.__signals[EEG_O2_A1_NO]

    def __load_cz2_a1(self):
        return self.__signals[EEG_CZ2_A1_NO]

    def __load_eog1_a2(self):
        return self.__signals[EOG1_A2_NO]

    def __load_eog2_a2(self):
        return self.__signals[EOG2_A2_NO]

    def __load_frequency(self):
        return self.__edf.getSampleFrequency(EEG_FPI_A2_NO) if self.__edf else 0

    def __load_samples_amount(self):
        return self.__edf.getNSamples()[EEG_FPI_A2_NO]

    def load_eeg(self):
        self.__load_fp2_a1()

    def load_eog(self):
        self.__load_eog1_a2()