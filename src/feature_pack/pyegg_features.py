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


# ----------------------------------------------DEPRECATED FUNCTIONS----------------------------------------------
@deprecated(reason="time-consuming")
def hurst(samples, **kwargs):
    """time-consuming"""
    return pyeeg.hurst(samples)


@deprecated(reason="Unresolved error: can not find module named 'embedded_sequence")
def largest_lyauponov_exponent(samples, **kwargs):
    """
    File "/usr/local/lib/python3.6/dist-packages/pyeeg-0.4.4-py3.6.egg/pyeeg/largest_lyauponov_exponent.py",
    line 75, in LLE
    ModuleNotFoundError: No module named 'embedded_sequence'
    """
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


@deprecated(reason="Unresolved error: 1-dimensional array given. Array must be two-dimensional")
def higuchi_fractal_dimension(samples, **kwargs):
    """
    numpy.linalg.linalg.LinAlgError:
    1-dimensional array given. Array must be two-dimensional
    """
    try:
        kmax = kwargs[KMAX]
    except KeyError:
        kmax = KMAX_DEFAULT
    return pyeeg.hfd(samples, Kmax=kmax)


@deprecated("Unresolved error: name 'bin_power' is not defined")
def spectral_entropy(samples, **kwargs):
    """
    NameError:
    name 'bin_power' is not defined
    """
    try:
        sampling_freq = kwargs[SAMPLING_FREQ]
        band = kwargs[BAND]
        power_ratio = kwargs[POWER_RATIO]
    except KeyError:
        sampling_freq = SAMPLING_FREQ_DEFAULT
        band = BAND_DEFAULT
        power_ratio = POWER_RATIO_DEFAULT
    return pyeeg.spectral_entropy(samples, Band=band, Fs=sampling_freq, Power_Ratio=power_ratio)


@deprecated(reason="Unresolved error: negative dimensions are not allowed")
def embedded_sequence(samples, **kwargs):
    """
    ValueError:
    negative dimensions are not allowed
    """
    try:
        embedding_dimension = kwargs[EMBEDDING_DIMENSION]  # 4
        embedding_lag = kwargs[EMBEDDING_LAG]  # 1
    except KeyError:
        embedding_dimension = EMBEDDING_DIMENSION_DEFAULT
        embedding_lag = EMBEDDING_LAG_DEFAULT
    return pyeeg.embed_seq(samples, tau=embedding_lag, embedding_dimension=embedding_dimension)


@deprecated(reason="Unresolved error: Arrays cannot be empty")
def detrended_fluctuation_analysis(samples, **kwargs):
    """
    numpy.linalg.linalg.LinAlgError:
    Arrays cannot be empty
    """
    try:
        average_value = kwargs[AVERAGE_VALUE]
        box_size_list = kwargs[BOX_SIZE_LIST]
    except KeyError:
        average_value = AVERAGE_VALUE_DEFAULT
        box_size_list = BOX_SIZE_LIST_DEFAULT
    return pyeeg.dfa(samples, Ave=average_value, L=box_size_list)
