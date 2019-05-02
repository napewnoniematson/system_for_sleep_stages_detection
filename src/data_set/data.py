import numpy as np
from src.utils.util import *
from collections import Counter
from src.model.examination import Examination
import src.file_system.file_manager as file


def split(data_set):
    """
    Split given calculated data set to features and classes / sleep stages

    :param data_set: calculated data set (_calculate_data_set_for_examination)
    :return: list of features and list of classes / sleep stages
    """
    labels = np.array([i[0] for i in data_set])
    features = np.array([i[1:] for i in data_set])
    return features, labels


def examinations_data_set(signal_titles, hypnogram_titles, features_callbacks, window, training, date, **kwargs):
    """
    Function provides generator which return calculated features for every examination

    :param signal_titles: list of examination file names
    :param hypnogram_titles: list of hypnogram file names
    :param features_callbacks:
    :param window: describe length of segments used in computing
    :param training: flag which determines where save calculated features (train directory-true, test directory-false)
    :param date: required to prepare correct path
    :param kwargs:
    :return: generator which return list of calculated features for every examination
    """
    path = TRAINING_FEATURE_FILE_DIR if training else TEST_FEATURE_FILE_DIR
    for s, h in zip(signal_titles, hypnogram_titles):
        examination = Examination(s, h)
        data_set = _calculate_data_set_for_examination(
            examination, features_callbacks, window, **kwargs
        )
        if data_set is not None:
            file.save(path.format(date, s[0:-4]), data_set)
            yield data_set


def _calculate_data_set_for_examination(examination, features_callbacks, window=WINDOW_DEFAULT,
                                        normalized=True, **kwargs):
    """
    Function calculates features for one examination

    :param examination: examination containing psg signals and hypnogram
    :param features_callbacks: list (signals is more then one) of list of callbacks to features function
    :param window: describe length of segments used in computing
    :param normalized: flag determines signal normalization usage
    :param kwargs: dictionary with extra parameters for feature functions
    :return: list of calculated features for one examination
    """
    # signals used for feature extraction
    signals = [
        examination.psg.load_eeg(),
        examination.psg.load_eog()
    ]
    # signal normalization if needed
    signals = _normalize_signals(signals, normalized=False)
    # sequence of classes registered by expert
    hypnogram = examination.hypnogram
    # check correctness of signals lengths
    if _check_correctness(signals, hypnogram, features_callbacks) is False:
        return None
    # preparing array of features
    begin, end = 0, _allowed_upper_bound(signals[0], window)
    calculated_features = []  # result container
    for i in range(begin, end, window):  # loop through signals (length)
        # [i:i+window] splitting signals by window non overlapping
        window_hypnogram = hypnogram[i:i + window]
        # looking for unwanted stages (undefined or movement)
        if _has_unknown(window_hypnogram):
            continue
        # calculate features for current part (window) of signal
        features = [_find_class(window_hypnogram)]
        for callbacks, signal in zip(features_callbacks, signals):
            for call in callbacks:
                features.append(
                    call(signal[i:i + window], **kwargs)
                )
        # print(features)
        # accumulate calculated features
        calculated_features.append(features)
    return calculated_features


def _check_correctness(signals, hypnogram, features_callbacks):
    """
    Function checks if every signal has its own list of features callbacks and if every signal's sample
    has its class / sleep stage

    :param signals: list of samples
    :param hypnogram: list of classes / sleep stages
    :param features_callbacks: list of callbacks to feature functions
    :return: boolean value of conditions mentioned in function description
    """
    if len(signals) != len(features_callbacks):
        return False
    for s in signals:
        if len(s) != len(hypnogram):
            return False
    return True


def _normalize_signals(signals, normalized=True):
    """
    Signal normalization: value is decreased by minimum value, and then difference is divided by peak to peak
    value (maximum - minimum). Mentioned calculation is repeated for every sample

    :param signals: list of sample
    :param normalized: flag, if true then signal will be normalized
    :return: list of normalized samples
    """
    return [(signal - min(signal) / signal.ptp()) for signal in signals] if normalized else signals


def _has_unknown(hypnogram):
    """
    Checks if given hypnogram has unknown class / sleep stage

    :param hypnogram: list of classes / sleep stages
    :return: True if at least one class / sleep stage is unknown
    """
    for h in hypnogram:
        if h == UNKNOWN:
            return True
    return False


def _allowed_upper_bound(samples, window):
    """
    Function finds signal length for computing based on samples and window. Function ensures that all segments will
    have the same length (window length), not less

    :param samples: list of samples
    :param window: size of window used in computing
    :return: length of signal
    """
    return len(samples) if len(samples) % window == 0 else len(samples) - window


def _find_class(hypnogram):
    """
    Function finds class / sleep stage for given hypnogram based on class appearances

    :param hypnogram: list of sleep stages
    :return: the most common class / sleep stage in hypnogram
    """
    return Counter(hypnogram).most_common()[0][0]
