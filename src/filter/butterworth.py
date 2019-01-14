from scipy.signal import butter, lfilter

import src.logger.logger as logger
from src.logger.messages import *


def butter_bandpass(low_cut, high_cut, fs, order=5):
    nyq = 0.5*fs  # Nyquist
    low = low_cut/nyq
    high = high_cut/nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, low_cut, high_cut, fs, order=5):
    # logger.info(BUTTERWORTH_FILTER_START)
    # logger.data_info(BUTTERWORTH_DATA_INFO.format(low_cut, high_cut, fs, order))
    b, a = butter_bandpass(low_cut, high_cut, fs, order=order)
    y = lfilter(b, a, data)
    # logger.info(BUTTERWORTH_FILTER_END)
    return y
