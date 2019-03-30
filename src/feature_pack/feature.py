import math
import numpy as np
import statistics as stat
import scipy.stats as sci_stat
import scipy.signal as signal
import scipy.fftpack as fft_pack
from src.utils.util import *
from src.filter.butterworth import butter_bandpass_filter
from deprecated import deprecated


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


def harmonic_mean(samples, **kwargs):
    inv_s = [1 / s for s in samples]
    return len(inv_s) / sum(inv_s)


def variance(samples, **kwargs):
    return stat.variance(samples)


def variation(samples, **kwargs):
    return sci_stat.variation(samples)


def spectrogram(samples, **kwargs):
    """
    ValueError:
    Error when checking input: expected flatten_1_input to have 2 dimensions, but got array with shape (30520, 1, 3)

    :param samples:
    :param kwargs:
    :return:
    """
    try:
        sampling_freq = kwargs[SAMPLING_FREQ]
    except KeyError:
        sampling_freq = SAMPLING_FREQ_DEFAULT
        return signal.spectrogram(samples, fs=sampling_freq)


def spectrogram2(samples, **kwargs):
    spec = spectrogram(samples, **kwargs)[2]
    return [i[0] for i in spec]


def median(samples, **kwargs):
    return stat.median(samples)


def energy(samples, **kwargs):
    # powered_abs = [abs(s) for s in samples.copy()]  # publication
    powered_abs = [abs(s) ** 2 for s in samples.copy()]  # internet
    return sum(powered_abs)


def spectrogram2_energy(samples, **kwargs):
    spec = spectrogram2(samples, **kwargs)
    return energy(spec, **kwargs)


def alpha_energy(samples, **kwargs):
    filtered = butter_bandpass_filter(samples.copy(), ALPHA_LOW, ALPHA_HIGH, 200)
    return energy(filtered)


def beta_energy(samples, **kwargs):
    filtered = butter_bandpass_filter(samples.copy(), BETA_LOW, BETA_HIGH, 200)
    return energy(filtered)


def theta_energy(samples, **kwargs):
    filtered = butter_bandpass_filter(samples.copy(), THETA_LOW, THETA_HIGH, 200)
    return energy(filtered)


def delta_energy(samples, **kwargs):
    filtered = butter_bandpass_filter(samples.copy(), DELTA_LOW, DELTA_HIGH, 200)
    return energy(filtered)


def alpha_energy_ratio(samples, **kwargs):
    return alpha_energy(samples) / (
            beta_energy(samples) + theta_energy(samples) + delta_energy(samples))


def beta_energy_ratio(samples, **kwargs):
    return beta_energy(samples) / (
            alpha_energy(samples) + theta_energy(samples) + delta_energy(samples))


def theta_energy_ratio(samples, **kwargs):
    return theta_energy(samples) / (
            alpha_energy(samples) + beta_energy(samples) + delta_energy(samples))


def delta_energy_ratio(samples, **kwargs):
    return delta_energy(samples) / (
            alpha_energy(samples) + beta_energy(samples) + theta_energy(samples))


def sig_pow(samples, **kwargs):
    e = energy(samples)
    return (1 / len(samples)) * e


def sig_pow_sk(samples, **kwargs):
    return sig_pow(samples) ** 0.5


def spectrogram2_sig_pow(samples, **kwargs):
    spec = spectrogram2(samples, **kwargs)
    return sig_pow(spec, **kwargs)


def spectrogram2_sig_pow_sk(samples, **kwargs):
    spec = spectrogram2(samples, **kwargs)
    return sig_pow_sk(spec, **kwargs)


def entropy(samples, **kwargs):
    x = [s ** 2 * math.log(s ** 2) for s in samples]  # publication
    # x = [s * math.log(s) for s in samples]  # internet
    return -(sum(x))


# FFT

def _value_of_complex(cx):
    return math.sqrt(cx.real * cx.real + cx.imag * cx.imag)


def _value_of_complexes(cxs):
    return [_value_of_complex(cx) for cx in cxs]


def _fft(samples):
    # return fft_pack.fft(samples)[1:len(samples) // 2] # maybe half of fft is enough
    return fft_pack.fft(samples.copy())


def _fft_window(samples):
    w = signal.blackman(len(samples))
    return _fft(samples * w)


def _fft_energy(samples, **kwargs):
    f = fft_pack.fft(samples.copy())
    # f = fft_pack.rfft(samples)  # maybe rfft is better
    # vc = _value_of_complexes(f)
    return energy(f)


def alpha_fft_energy(samples, **kwargs):
    filtered = butter_bandpass_filter(samples.copy(), ALPHA_LOW, ALPHA_HIGH, 200)
    return _fft_energy(filtered)


def beta_fft_energy(samples, **kwargs):
    filtered = butter_bandpass_filter(samples.copy(), BETA_LOW, BETA_HIGH, 200)
    return _fft_energy(filtered)


def theta_fft_energy(samples, **kwargs):
    filtered = butter_bandpass_filter(samples.copy(), THETA_LOW, THETA_HIGH, 200)
    return _fft_energy(filtered)


def delta_fft_energy(samples, **kwargs):
    filtered = butter_bandpass_filter(samples.copy(), DELTA_LOW, DELTA_HIGH, 200)
    return _fft_energy(filtered)


def alpha_fft_energy_ratio(samples, **kwargs):
    return alpha_fft_energy(samples) / (
            beta_fft_energy(samples) + theta_fft_energy(samples) + delta_fft_energy(samples))


def beta_fft_energy_ratio(samples, **kwargs):
    return beta_fft_energy(samples) / (
            alpha_fft_energy(samples) + theta_fft_energy(samples) + delta_fft_energy(samples))


def theta_fft_energy_ratio(samples, **kwargs):
    return theta_fft_energy(samples) / (
            alpha_fft_energy(samples) + beta_fft_energy(samples) + delta_fft_energy(samples))


def delta_fft_energy_ratio(samples, **kwargs):
    return delta_fft_energy(samples) / (
            alpha_fft_energy(samples) + beta_fft_energy(samples) + theta_fft_energy(samples))


def callbacks():
    return [min_peak, max_peak, variance, spectrogram, mean, root_mean_square, median]
