import pyedflib
import numpy as np

EEG_FPZ_CZ = 0
EEG_PZ_OZ = 1
EOG = 2
IMPORTANT_SIGNALS = [EEG_FPZ_CZ, EOG]
IMPORTANT_SIGNALS_AMOUNT = len(IMPORTANT_SIGNALS)
EEG_EOG_FREQ = 100


class PSG:
    def __init__(self, path):
        self.__edf = pyedflib.EdfReader(path)
        self.__read_signals()

    def __read_signals(self):
        self.__signals = np.zeros((IMPORTANT_SIGNALS_AMOUNT, self.__edf.getNSamples()[0]))
        for i in range(IMPORTANT_SIGNALS_AMOUNT):
            self.__signals[i, :] = self.__edf.readSignal(IMPORTANT_SIGNALS[i])

    def load_eeg(self):
        return self.__signals[0]

    def load_eog(self):
        return self.__signals[1]


