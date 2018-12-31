import src.loader.aasm as aasm
from src.loader.eeg import EEG
from src.utils.util import *


class Examination:
    def __init__(self, subject_title):
        self.eeg = EEG(UNIVERSITE_DE_MONS.format(subject_title))
        self.hypnogram = aasm.load_extended_data(UNIVERSITE_DE_MONS_HYPNOGRAM_AASM.format(subject_title))
