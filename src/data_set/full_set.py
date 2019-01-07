from src.utils.util import *
from src.model.examination import Examination
from collections import Counter


def load_examinations(examination_titles):
    return [Examination(title) for title in examination_titles]


def check_correctness(signals, hypnogram):
    for s in signals:
        if len(s) != len(hypnogram):
            raise Exception("Hypnogram and signal length are not the same")


def examinations_data_set(examinations, features_callbacks, window, **kwargs):
    data_set = []
    for examination in examinations:
        data_set.append(
            _calculate_data_set_for_examination(
                examination, features_callbacks, window, **kwargs
            )
        )
    return data_set


def _calculate_data_set_for_examination(examination, features_callbacks, window=WINDOW_DEFAULT, **kwargs):
    signals = [examination.psg.load_fpi_a2(), examination.psg.load_eog1_a2()]
    hypnogram = examination.hypnogram
    check_correctness(signals, hypnogram)
    if len(signals) != len(features_callbacks):
        raise Exception("Chuj wam w dupe")
    begin = 0
    end = _allowed_upper_bound(signals[0], window)
    calculated_features = []
    for i in range(begin, end, window):
        window_hypnogram = hypnogram[i:i + window]
        if _has_unknown(window_hypnogram):
            continue
        label = _find_class(window_hypnogram)
        feat = []
        for callbacks, signal in zip(features_callbacks, signals):
            for func in callbacks:
                feat.append(func(signal[i:i+window], **kwargs))

        calculated_features.append(
            (
                _find_class(window_hypnogram),
                feat
            )
        )
    return calculated_features


def _has_unknown(hypnogram):
    for h in hypnogram:
        if h is UNKNOWN:
            return True
    return False


def _allowed_upper_bound(samples, window):
    return len(samples) if len(samples) % window == 0 else len(samples) - window


def _find_class(hypnogram):
    return Counter(hypnogram).most_common()[0][0]


def _calculate_features_for_window(samples, callbacks, **kwargs):
    return [c(samples, **kwargs) for c in callbacks]
