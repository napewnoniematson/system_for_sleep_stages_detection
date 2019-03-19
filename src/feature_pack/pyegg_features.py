import pyeeg
from src.utils.util import *


def hurst(samples, **kwargs):
    """really really bad"""
    return pyeeg.hurst(samples)


def bin_power(samples, **kwargs):
    try:
        sampling_freq = kwargs[SAMPLING_FREQ]
        band = kwargs[BAND]
    except KeyError:
        sampling_freq = SAMPLING_FREQ_DEFAULT
        band = BAND_DEFAULT
    return pyeeg.bin_power(samples, Band=band, Fs=sampling_freq)


def largest_lyauponov_exponent(samples, **kwargs):
    try:
        embedding_dimension = kwargs[EMBEDDING_DIMENSION]
        embedding_lag = kwargs[EMBEDDING_LAG]
        sampling_freq = kwargs[SAMPLING_FREQ]
        mean_period = kwargs[MEAN_PERIOD]
    except KeyError:
        embedding_dimension = EMBEDDING_DIMENSION_DEFAULT
        embedding_lag = EMBEDDING_LAG_DEFAULT
        sampling_freq = SAMPLING_FREQ_DEFAULT
        mean_period = MEAN_PERIOD_DEFAULT
    return pyeeg.LLE(samples, tau=embedding_lag, n=embedding_dimension, T=mean_period, fs=sampling_freq)


def hjorth(samples, **kwargs):
    try:
        differential_sequence = kwargs[DIFF_SEQ]
    except KeyError:
        differential_sequence = DIFF_SEQ_DEFAULT
    return pyeeg.hjorth(samples, D=differential_sequence)


def higuchi_fractal_dimension(samples, **kwargs):
    try:
        kmax = kwargs[KMAX]
    except KeyError:
        kmax = KMAX_DEFAULT
    return pyeeg.hfd(samples, Kmax=kmax)


def petrosian_fractal_dimension(samples, **kwargs):
    try:
        differential_sequence = kwargs[DIFF_SEQ]
    except KeyError:
        differential_sequence = DIFF_SEQ_DEFAULT
    return pyeeg.pfd(samples, D=differential_sequence)


def fisher_info(samples, **kwargs):
    try:
        embedding_dimension = kwargs[EMBEDDING_DIMENSION]  # 4
        embedding_lag = kwargs[EMBEDDING_LAG]  # 1
        singular_values_of_matrix = kwargs[SINGULAR_VALUES]
    except KeyError:
        embedding_dimension = EMBEDDING_DIMENSION_DEFAULT
        embedding_lag = EMBEDDING_LAG_DEFAULT
        singular_values_of_matrix = SINGULAR_VALUES_DEFAULT
    return pyeeg.fisher_info(samples, Tau=embedding_lag, DE=embedding_dimension, W=singular_values_of_matrix)


def permutation_entropy(samples, **kwargs):
    try:
        permutation_order = kwargs[PERMUTATION_ORDER]  # 5
        embedding_lag = kwargs[EMBEDDING_LAG]  # 1
    except KeyError:
        permutation_order = PERMUTATION_ORDER_DEFAULT
        embedding_lag = EMBEDDING_LAG_DEFAULT
    return pyeeg.permutation_entropy(samples, n=permutation_order, tau=embedding_lag)


def spectral_entropy(samples, **kwargs):
    try:
        sampling_freq = kwargs[SAMPLING_FREQ]
        band = kwargs[BAND]
        power_ratio = kwargs[POWER_RATIO]
    except KeyError:
        sampling_freq = SAMPLING_FREQ_DEFAULT
        band = BAND_DEFAULT
        power_ratio = POWER_RATIO_DEFAULT
    return pyeeg.spectral_entropy(samples, Band=band, Fs=sampling_freq, Power_Ratio=power_ratio)


def svd_entropy(samples, **kwargs):
    try:
        embedding_dimension = kwargs[EMBEDDING_DIMENSION]  # 4
        embedding_lag = kwargs[EMBEDDING_LAG]  # 1
        singular_values_of_matrix = kwargs[SINGULAR_VALUES]
    except KeyError:
        embedding_dimension = EMBEDDING_DIMENSION_DEFAULT
        embedding_lag = EMBEDDING_LAG_DEFAULT
        singular_values_of_matrix = SINGULAR_VALUES_DEFAULT
    return pyeeg.svd_entropy(samples, Tau=embedding_lag, DE=embedding_dimension, W=singular_values_of_matrix)


def embedded_sequence(samples, **kwargs):
    try:
        embedding_dimension = kwargs[EMBEDDING_DIMENSION]  # 4
        embedding_lag = kwargs[EMBEDDING_LAG]  # 1
    except KeyError:
        embedding_dimension = EMBEDDING_DIMENSION_DEFAULT
        embedding_lag = EMBEDDING_LAG_DEFAULT
    return pyeeg.embed_seq(samples, tau=embedding_lag, n=embedding_dimension)


def detrended_fluctuation_analysis(samples, **kwargs):
    try:
        average_value = kwargs[AVERAGE_VALUE]
        box_size_list = kwargs[BOX_SIZE_LIST]
    except KeyError:
        average_value = AVERAGE_VALUE_DEFAULT
        box_size_list = BOX_SIZE_LIST_DEFAULT
    return pyeeg.dfa(samples, Ave=average_value, L=box_size_list)
