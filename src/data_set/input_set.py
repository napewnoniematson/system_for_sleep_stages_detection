from src.utils.util import *
from src.model.unversite_de_mons.examination import Examination as UDMExamination
from src.model.physionet_sleep_cassette.examination import Examination as PhysioExamination
from collections import Counter


def load_udm_examinations(examination_titles):
    return [UDMExamination(title) for title in examination_titles]


def load_physio_examinations(signal_titles, hypnogram_titles):
    return [PhysioExamination(s, h) for s, h in zip(signal_titles, hypnogram_titles)]


def check_correctness(signals, hypnogram):
    for s in signals:
        if len(s) != len(hypnogram):
            raise Exception("Hypnogram and signal length are not the same")


def examinations_data_set(examinations, features_callbacks, window, **kwargs):
    data_set = []
    temp_ctr = 1
    for examination in examinations:
        print("Examination {}".format(temp_ctr))
        data_set.append(
            _calculate_data_set_for_examination(
                examination, features_callbacks, window, **kwargs
            )
        )
        temp_ctr += 1
    return data_set


def _calculate_data_set_for_examination(examination, features_callbacks, window=WINDOW_DEFAULT,
                                        normalized=True, **kwargs):
    signals = [
        examination.psg.load_eeg(),
        examination.psg.load_eog()
    ]
    if normalized:
        signals = [(signal - min(signal) / signal.ptp()) for signal in signals]
        # import src.feature_pack.feature as f
        # signals = [(signal / f.mean(signal)) for signal in signals]
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
        feat = []
        for callbacks, signal in zip(features_callbacks, signals):
            for func in callbacks:
                feat.append(func(signal[i:i + window], **kwargs))

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
