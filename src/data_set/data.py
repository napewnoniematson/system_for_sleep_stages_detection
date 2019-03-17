import numpy as np
from src.utils.util import *
from collections import Counter
from src.model.examination import Examination
import src.file_system.file_manager as file


def split(data_set):
    labels = np.array([i[0] for i in data_set])
    features = np.array([i[1:] for i in data_set])
    return features, labels


def examinations_data_set(signal_titles, hypnogram_titles, features_callbacks, window, training, date, **kwargs):
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
    # signals used for feature extraction
    signals = [
        examination.psg.load_eeg(),
        examination.psg.load_eog()
    ]
    # signal normalization if needed
    signals = _normalize_signals(signals, normalized)
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
        # accumulate calculated features
        calculated_features.append(features)
    return calculated_features


def _check_correctness(signals, hypnogram, features_callbacks):
    if len(signals) != len(features_callbacks):
        return False
    for s in signals:
        if len(s) != len(hypnogram):
            return False
    return True


def _normalize_signals(signals, normalized=True):
    return [(signal - min(signal) / signal.ptp()) for signal in signals] if normalized else signals


def _has_unknown(hypnogram):
    for h in hypnogram:
        if h == UNKNOWN:
            return True
    return False


def _allowed_upper_bound(samples, window):
    return len(samples) if len(samples) % window == 0 else len(samples) - window


def _find_class(hypnogram):
    return Counter(hypnogram).most_common()[0][0]
