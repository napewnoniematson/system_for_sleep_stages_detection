from enum import Enum

import src.loader.universite_de_mons_data.hypnogram.aasm as aasm
import src.loader.universite_de_mons_data.hypnogram.rk as rk
from src.loader.universite_de_mons_data.edf.eeg import EEG
from src.utils.util import *


class Option(Enum):
    AASM = 0
    RK = 1


class Examination:
    def __init__(self, subject_no, option=Option.AASM):
        self.eeg = EEG(UNIVERSITE_DE_MONS.format(subject_no))
        self.hypnogram = self.__load_hypnogram(subject_no, option)

    def __load_hypnogram(self, subject_no, option):
        hypnogram = []
        if option is Option.AASM:
            hypnogram = aasm.load(UNIVERSITE_DE_MONS_HYPNOGRAM_AASM.format(subject_no))
        elif option is Option.RK:
            hypnogram = rk.load(UNIVERSITE_DE_MONS_HYPNOGRAM_RK.format(subject_no))
        return hypnogram
