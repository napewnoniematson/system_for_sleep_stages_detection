import math
import statistics as stat
import scipy.signal as signal
import scipy.fftpack as fft_pack
from src.utils.util import *
from src.filter.butterworth import butter_bandpass_filter


def root_mean_square(samples, **kwargs):
    size = len(samples)
    sum_pow = 0
    for y in samples:
        sum_pow += y * y
    return math.sqrt(sum_pow / size)


def max_peak(samples, **kwargs):
    return max(samples)


def min_peak(samples, **kwargs):
    return min(samples)


def mean(samples, **kwargs):
    return stat.mean(samples)


def variance(samples, **kwargs):
    return stat.variance(samples)


def spectrogram(samples, **kwargs):
    try:
        sampling_freq = kwargs[SAMPLING_FREQ]
    except KeyError:
        sampling_freq = SAMPLING_FREQ_DEFAULT
    return signal.spectrogram(samples, fs=sampling_freq)


def value(samples, **kwargs):
    length = len(samples)
    index = int(length / 2) + 1
    return samples[index]


def median(samples, **kwargs):
    return stat.median(samples)


# todo test
def mean_normalized(samples, **kwargs):
    mean_val = mean(samples)
    min_val = min_peak(samples)
    max_val = max_peak(samples)
    return (mean_val - min_val) / (max_val - min_val)


def sum_of_fft(samples, **kwargs):
    return sum(fft_pack.fft(samples))


def alpha_energy(samples, **kwargs):
    filtered = butter_bandpass_filter(samples, ALPHA_LOW, ALPHA_HIGH, 200)
    return sum_of_fft(filtered)


def beta_energy(samples, **kwargs):
    filtered = butter_bandpass_filter(samples, BETA_LOW, BETA_HIGH, 200)
    return sum_of_fft(filtered)


def theta_energy(samples, **kwargs):
    filtered = butter_bandpass_filter(samples, THETA_LOW, THETA_HIGH, 200)
    return sum_of_fft(filtered)


def delta_energy(samples, **kwargs):
    filtered = butter_bandpass_filter(samples, DELTA_LOW, DELTA_HIGH, 200)
    return sum_of_fft(filtered)


def alpha_energy_ratio(samples, **kwargs):
    return alpha_energy(samples) / (beta_energy(samples) + theta_energy(samples) + delta_energy(samples))


def beta_energy_ratio(samples, **kwargs):
    return beta_energy(samples) / (alpha_energy(samples) + theta_energy(samples) + delta_energy(samples))


def theta_energy_ratio(samples, **kwargs):
    return theta_energy(samples) / (alpha_energy(samples) + beta_energy(samples) + delta_energy(samples))


def delta_energy_ratio(samples, **kwargs):
    return delta_energy(samples) / (alpha_energy(samples) + beta_energy(samples) + theta_energy(samples))


def callbacks():
    return [value, min_peak, max_peak, variance, spectrogram, mean, root_mean_square, median, mean_normalized,
            sum_of_fft]
