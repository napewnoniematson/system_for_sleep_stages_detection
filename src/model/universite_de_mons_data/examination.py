from enum import Enum

import src.loader.universite_de_mons_data.hypnogram.aasm as aasm
import src.loader.universite_de_mons_data.hypnogram.rk as rk
from src.loader.universite_de_mons_data.edf.eeg import EEG
from src.utils.util import *


class HypnogramType(Enum):
    AASM = 0
    RK = 1


class Examination:
    def __init__(self, subject_title, hypnogram_type=HypnogramType.AASM):
        self.eeg = EEG(UNIVERSITE_DE_MONS.format(subject_title))
        self.hypnogram = self.__load_hypnogram(subject_title, hypnogram_type)

    def __load_hypnogram(self, subject_title, hypnogram_type):
        hypnogram = []
        if hypnogram_type is HypnogramType.AASM:
            hypnogram = aasm.load(UNIVERSITE_DE_MONS_HYPNOGRAM_AASM.format(subject_title))
        elif hypnogram_type is HypnogramType.RK:
            hypnogram = rk.load(UNIVERSITE_DE_MONS_HYPNOGRAM_RK.format(subject_title))
        return hypnogram
