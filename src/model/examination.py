from src.loader.physionet_sleep_cassette.rk import RK as p_RK
from src.loader.physionet_sleep_cassette.psg import PSG as p_PSG
from src.loader.unversite_de_mons.rk import RK as u_RK
from src.loader.unversite_de_mons.psg import PSG as u_PSG
from src.utils.util import *


class Examination:
    def __init__(self, signal_title, hypnogram_title):
        if 'subject' in signal_title:
            self.psg = u_PSG(DE_MONS_DIR_LOAD.format(signal_title))
            self.hypnogram = u_RK(DE_MONS_DIR_LOAD.format(hypnogram_title)).hypnogram
        else:
            self.psg = p_PSG(PHYSIONET_DIR_LOAD.format(signal_title))
            self.hypnogram = p_RK(PHYSIONET_DIR_LOAD.format(hypnogram_title)).hypnogram
