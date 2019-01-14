import src.loader.unversite_de_mons.aasm as aasm
from src.loader.unversite_de_mons.psg import PSG
from src.utils.util import *


class Examination:
    def __init__(self, subject_title):
        self.psg = PSG(UNIVERSITE_DE_MONS.format(subject_title))
        self.hypnogram = aasm.load(UNIVERSITE_DE_MONS_HYPNOGRAM_AASM.format(subject_title))
