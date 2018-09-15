import matplotlib.pyplot as plt

from src.model.universite_de_mons_data.examination import Examination, HypnogramType
from src.utils.util import *


class Controller:
    SUBJECT_NO = 16

    def __init__(self):
        self.examination = Examination(self.SUBJECT_NO)
        self.fpi_a2 = self.examination.eeg.load_fpi_a2()

    def run(self):
        plt.plot(self.examination.hypnogram["data"])
        plt.show()

