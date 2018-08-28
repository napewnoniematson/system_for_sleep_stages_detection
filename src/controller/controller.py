import matplotlib.pyplot as plt

import src.loader.universite_de_mons_data.hypnogram.aasm as aasm
import src.loader.universite_de_mons_data.hypnogram.rk as rk
from src.loader.universite_de_mons_data.edf.eeg import EEG
from src.utils.util import *


class Controller:

    SUBJECT_NO = 16

    def __init__(self):
        self.eeg = EEG(UNIVERSITE_DE_MONS.format(self.SUBJECT_NO))
        self.aasm = aasm.load(UNIVERSITE_DE_MONS_HYPNOGRAM_AASM.format(self.SUBJECT_NO))
        self.rk = rk.load(UNIVERSITE_DE_MONS_HYPNOGRAM_AASM.format(self.SUBJECT_NO))
        self.fpi_a2 = self.eeg.load_fpi_a2()

    def run(self):
        plt.plot(self.fpi_a2[30000:32000])
        plt.show()
