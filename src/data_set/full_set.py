from src.decorator.timer import timer
from src.utils.util import *
from src.model.stage_mode import StageMode
from src.model.examination import Examination
from deprecated import deprecated
from collections import Counter


@timer
def load_examinations(examination_titles):
    return [Examination(title) for title in examination_titles]


def all_stages_to_rem_vs_non_rem(hypnogram):
    return ['1' if stage is '1' else '0' for stage in hypnogram]


def hypnogram_by_stage(hypnogram, stage_mode):
    return all_stages_to_rem_vs_non_rem(hypnogram) if StageMode.REM_VS_NREM.value == stage_mode else hypnogram


def check_correctness(samples, hypnogram):
    if len(samples) != len(hypnogram):
        raise Exception("Hypnogram and signal length are not the same")


@timer
def examinations_data_set(examinations, features_callbacks, window_width,
                          stage_mode, **kwargs):
    data_set = []
    for examination in examinations:
        data_set.append(
            _calculate_data_set_for_examination(
                examination, features_callbacks, window_width, stage_mode, **kwargs
            )
        )
    return data_set


def _calculate_data_set_for_examination(examination, features_callbacks, window_width=WINDOW_WIDTH_DEFAULT,
                                        stage_mode=STAGE_MODE_VALUE_DEFAULT, **kwargs):
    samples = examination.eeg.load_fpi_a2()
    hypnogram = hypnogram_by_stage(examination.hypnogram, stage_mode)
    check_correctness(samples, hypnogram)

    begin = 0
    end = _allowed_upper_bound(samples, window_width)
    calculated_features = []
    for i in range(begin, end, window_width):
        window_samples = samples[i:i + window_width]
        window_hypnogram = hypnogram[i:i + window_width]
        calculated_features.append(
            (
                _find_class(window_hypnogram),
                _calculate_features_for_window(window_samples, features_callbacks, **kwargs)
            )
        )
    return calculated_features


def _allowed_upper_bound(samples, window_width):
    return len(samples) if len(samples) % window_width == 0 else len(samples) - window_width


def _find_class(window_hypnogram):
    return Counter(window_hypnogram).most_common()[0][0]


def _calculate_features_for_window(window_samples, features_callbacks, **kwargs):
    calculated_features = []
    for feature_callback in features_callbacks:
        calculated_features.append(feature_callback(window_samples, **kwargs))
    return calculated_features


@deprecated(reason=DEPRECATED_REASON)
def prepare_data_set_old(examination_titles, features_callbacks, window_width,
                         stage_mode, **kwargs):
    data_set = []
    for title in examination_titles:
        # load examination
        examination = Examination(title)
        eeg = examination.eeg.load_fpi_a2()
        stages_indices = _get_stages_indices(examination)
        stages = _convert_indices_to_values(eeg, stages_indices)
        data_set.append(
            _prepare_data_set_for_one_examination(
                stages, features_callbacks, window_width, stage_mode, **kwargs
            )
        )
    return data_set


# =================================================DEPRECATED==========================================================
@deprecated(reason=DEPRECATED_REASON)
def flatten_4_level_data_set(data_set):
    temp = []
    for examination in data_set:
        for i in examination:
            for j in i:
                for x in j:
                    temp.append(x)
    return temp


@deprecated(reason=DEPRECATED_REASON)
def _prepare_data_set_for_one_examination(stages, features_callbacks, window_width=WINDOW_WIDTH_DEFAULT,
                                          stage_mode=STAGE_MODE_VALUE_DEFAULT, **kwargs):
    class_nos = stages_to_class_nos(stages, stage_mode)
    data_set = []
    for value, class_no in zip(stages.values(), class_nos):
        data_set.append(
            _prepare_data_set_for_one_stage(
                value, class_no, features_callbacks, window_width, **kwargs
            )
        )
    return data_set


@deprecated(reason=DEPRECATED_REASON)
def _prepare_data_set_for_one_stage(stage, class_no, features_callbacks,
                                    window_width=WINDOW_WIDTH_DEFAULT, **kwargs):
    calculated_features_set = []
    for samples in stage:
        calculated_features = _calculate_features_overlapping(samples, features_callbacks, window_width, **kwargs)
        calculated_features_set.append(calculated_features)
    data_set = [[(class_no, cf) for cf in calculated_features] for calculated_features in calculated_features_set]
    return data_set


@deprecated(reason=DEPRECATED_REASON)
def _calculate_features_overlapping(samples, features_callbacks, window_width=WINDOW_WIDTH_DEFAULT, **kwargs):
    if window_width % 2 is 0:
        raise Exception(WINDOW_SIZE_EXCEPTION_MESSAGE)
    pre = int(window_width / 2)
    post = int(window_width - pre)
    samples_amount = len(samples)
    calculated_features = []
    for i in range(samples_amount):
        if i < pre or i > samples_amount - post:
            continue
        window_samples = samples[i - pre:i + post]
        calculated_features.append(
            _calculate_features_for_window(window_samples, features_callbacks, **kwargs)
        )
    return calculated_features


@deprecated(reason=DEPRECATED_REASON)
def stages_to_class_nos(stages, stage_mode=STAGE_MODE_VALUE_DEFAULT):
    keys = stages.keys()
    class_nos = []
    if StageMode.REM_VS_NREM.value == stage_mode:
        class_nos = [1 if k is 'rem' else 0 for k in keys]
    elif StageMode.ALL_STAGES.value is stage_mode:
        class_nos = [i for i in range(len(keys))]
    else:
        pass
    return class_nos


@deprecated(reason=DEPRECATED_REASON)
def _get_stages_indices(examination):
    return {key: value for key, value in examination.hypnogram.items()
            if not (key is 'title' or key is 'data')}


@deprecated(reason=DEPRECATED_REASON)
def _convert_indices_to_values(eeg, stages_indices):
    return {key: [eeg[indices[0]:indices[1]] for indices in value] for key, value in stages_indices.items()}
