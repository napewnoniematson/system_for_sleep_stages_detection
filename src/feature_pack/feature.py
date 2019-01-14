import math
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
    try:
        sampling_freq = kwargs[SAMPLING_FREQ]
    except KeyError:
        sampling_freq = SAMPLING_FREQ_DEFAULT
        return signal.spectrogram(samples, fs=sampling_freq)


def median(samples, **kwargs):
    return stat.median(samples)


def energy(samples, **kwargs):
    # powered_abs = [abs(s) for s in samples.copy()]  # publication
    powered_abs = [abs(s) ** 2 for s in samples.copy()]  # internet
    return sum(powered_abs)


def sig_pow(samples, **kwargs):
    e = energy(samples)
    return (1 / len(samples)) * e


def sig_pow_sk(samples, **kwargs):
    return sig_pow(samples) ** 0.5


def entropy(samples, **kwargs):
    x = [s**2 * math.log(s**2) for s in samples]  # publication
    # x = [s * math.log(s) for s in samples]  # internet
    return -(sum(x))


# FFT

def _value_of_complex(cx):
    return math.sqrt(cx.real * cx.real + cx.imag * cx.imag)


def _value_of_complexes(cxs):
    return [_value_of_complex(cx) for cx in cxs]


def _fft(samples):
    # return fft_pack.fft(samples)[1:len(samples) // 2] # maybe half of fft is enough
    return fft_pack.fft(samples)


def _fft_window(samples):
    w = signal.blackman(len(samples))
    return _fft(samples * w)


def _fft_energy(samples, **kwargs):
    f = fft_pack.fft(samples)
    # f = fft_pack.rfft(samples)  # maybe rfft is better
    # vc = _value_of_complexes(f)
    return energy(f)


def alpha_energy(samples, **kwargs):
    filtered = butter_bandpass_filter(samples.copy(), ALPHA_LOW, ALPHA_HIGH, 200)
    return _fft_energy(filtered)


def beta_energy(samples, **kwargs):
    filtered = butter_bandpass_filter(samples.copy(), BETA_LOW, BETA_HIGH, 200)
    return _fft_energy(filtered)


def theta_energy(samples, **kwargs):
    filtered = butter_bandpass_filter(samples.copy(), THETA_LOW, THETA_HIGH, 200)
    return _fft_energy(filtered)


def delta_energy(samples, **kwargs):
    filtered = butter_bandpass_filter(samples.copy(), DELTA_LOW, DELTA_HIGH, 200)
    return _fft_energy(filtered)


def alpha_energy_ratio(samples, **kwargs):
    return alpha_energy(samples) / (beta_energy(samples) + theta_energy(samples) + delta_energy(samples))


def beta_energy_ratio(samples, **kwargs):
    return beta_energy(samples) / (alpha_energy(samples) + theta_energy(samples) + delta_energy(samples))


def theta_energy_ratio(samples, **kwargs):
    return theta_energy(samples) / (alpha_energy(samples) + beta_energy(samples) + delta_energy(samples))


def delta_energy_ratio(samples, **kwargs):
    return delta_energy(samples) / (alpha_energy(samples) + beta_energy(samples) + theta_energy(samples))


def callbacks():
    return [min_peak, max_peak, variance, spectrogram, mean, root_mean_square, median]
