from src.loader.physionet_sleep_cassette.rk import RK
from src.loader.physionet_sleep_cassette.psg import PSG
from src.utils.util import *


class Examination:
    def __init__(self, signal_title, hypnogram_title):
        self.psg = PSG(PHYSIONET_DIR_LOAD.format(signal_title))
        self.hypnogram = RK(PHYSIONET_DIR_LOAD.format(hypnogram_title))
