import math
import statistics as stat
import scipy.signal as signal

class Provider:

    def __init__(self, eeg, data_len):
        self._data_set = [[] for _ in range(data_len)]
        self._eeg = eeg
        # self._data_set[0].append("rms")
        # self._data_set[0].append("var")
        # self._data_set[0].append("species")
        self._current_index = 0

    def update(self, range_array, phase_no):
        for w in range_array:
            index = self._current_index
            temp_signal = self._eeg[w[0]:w[1]]
            rms = root_mean_square(temp_signal)
            self._data_set[index].append(rms)
            var = variance(temp_signal)
            self._data_set[index].append(var)
            temp_mean = mean(temp_signal)
            self._data_set[index].append(temp_mean)
            temp_max_peak = max_peak(temp_signal)
            self._data_set[index].append(temp_max_peak)
            """0 - wake"""
            self._data_set[index].append(phase_no)
            self._index_increase()

    def _index_increase(self):
        self._current_index = self._current_index + 1

    def get_data_set(self):
        return self._data_set


def root_mean_square(samples):
    size = len(samples)
    sum_pow = 0
    for y in samples:
        sum_pow += y*y
    return math.sqrt(sum_pow/size)


def max_peak(samples):
    max_val = 0
    for y in samples:
        if y > max_val:
            max_val = y
    return max_val


def mean(samples):
    return stat.mean(samples)


def variance(samples):
    return stat.variance(samples)


def spectrogram(samples, sampling_freq):
    return signal.spectrogram(samples, fs=sampling_freq)