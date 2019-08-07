import math
import statistics as stat


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


def median(samples, **kwargs):
    return stat.median(samples)


def energy(samples, **kwargs):
    # powered_abs = [abs(s) for s in samples.copy()]  # publication
    powered_abs = [abs(s) ** 2 for s in samples.copy()]  # internet
    return sum(powered_abs)


def sig_pow(samples, **kwargs):
    e = energy(samples)
    return (1 / len(samples)) * e
