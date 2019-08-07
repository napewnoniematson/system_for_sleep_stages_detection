import pyeeg
from src.utils.util import *
from deprecated import deprecated


def _bin_power(samples, **kwargs):
    """ValueError:
     Error when checking input: expected flatten_1_input to have 2 dimensions,
     but got array with shape (30520, 1, 2, 4)

     Only for local usage: power functions / power ratio functions
     """
    try:
        sampling_freq = kwargs[SAMPLING_FREQ]
        band = kwargs[BAND]
    except KeyError:
        sampling_freq = SAMPLING_FREQ_DEFAULT
        band = BAND_DEFAULT
    return pyeeg.bin_power(samples, Band=band, Fs=sampling_freq)


def power_delta(samples, **kwargs):
    return _bin_power(samples, **kwargs)[0][0]


def power_delta_ratio(samples, **kwargs):
    return _bin_power(samples, **kwargs)[1][0]


def power_theta(samples, **kwargs):
    return _bin_power(samples, **kwargs)[0][1]


def power_theta_ratio(samples, **kwargs):
    return _bin_power(samples, **kwargs)[1][1]


def power_alpha(samples, **kwargs):
    return _bin_power(samples, **kwargs)[0][2]


def power_alpha_ratio(samples, **kwargs):
    return _bin_power(samples, **kwargs)[1][2]


def power_beta(samples, **kwargs):
    return _bin_power(samples, **kwargs)[0][3]


def power_beta_ratio(samples, **kwargs):
    return _bin_power(samples, **kwargs)[0][3]


def _hjorth(samples, **kwargs):
    """
    ValueError:
    Error when checking input: expected flatten_1_input to have 2 dimensions, but got array with shape (30520, 1, 2)

    Only for local usage: mobility / complexity
    """
    try:
        differential_sequence = kwargs[DIFF_SEQ]
    except KeyError:
        differential_sequence = DIFF_SEQ_DEFAULT
    return pyeeg.hjorth(samples, D=differential_sequence)


def hjorth_mobility(samples, **kwargs):
    return _hjorth(samples, **kwargs)[0]


def hjorth_complexity(samples, **kwargs):
    return _hjorth(samples, **kwargs)[1]


def petrosian_fractal_dimension(samples, **kwargs):
    try:
        differential_sequence = kwargs[DIFF_SEQ]
    except KeyError:
        differential_sequence = DIFF_SEQ_DEFAULT
    return pyeeg.pfd(samples, D=differential_sequence)


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
