import math
import statistics as stat
import scipy.signal as signal
from src.utils.util import *


def root_mean_square(samples, **kwargs):
    size = len(samples)
    sum_pow = 0
    for y in samples:
        sum_pow += y * y
    return math.sqrt(sum_pow / size)


def max_peak(samples, **kwargs):
    max_val = -3000
    for y in samples:
        if y > max_val:
            max_val = y
    if max_val <= -3000:
        raise ValueError
    return max_val


def min_peak(samples, **kwargs):
    min_val = 3000
    for y in samples:
        if y < min_val:
            min_val = y
    if min_val >= 3000:
        raise ValueError
    return min_val


def mean(samples, **kwargs):
    return stat.mean(samples)


def variance(samples, **kwargs):
    return stat.variance(samples)


def spectrogram(samples, **kwargs):
    try:
        sampling_freq = kwargs[SAMPLING_FREQ]
    except KeyError:
        sampling_freq = 200
    return signal.spectrogram(samples, fs=sampling_freq)


def value(samples, **kwargs):
    length = len(samples)
    index = int(length / 2) + 1
    return samples[index]


def median(samples, **kwargs):
    return stat.median(samples)


def callbacks():
    return [value, min_peak, max_peak, variance, spectrogram, mean, root_mean_square, median]
