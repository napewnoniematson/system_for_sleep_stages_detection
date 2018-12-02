from enum import Enum
from src.utils.util import *
import src.feature.provider as features
from src.model.universite_de_mons_data.examination import Examination, HypnogramType

class StagesMode(Enum):
    REM_VS_NO_REM = 0,
    ALL_STAGES = 1


class TrainController:
    def __init__(self):
        self.features_callbacks = features.callbacks()
        self.features_titles = [c.__name__ for c in self.features_callbacks]
        self.examinations = ["subject{}".format(i) for i in range(20)]
        self.window_sizes = [("{}s".format(i * 10 + 5), i * 10 + 5) for i in range(5)]

    def on_train_button_click(self, input_data):
        checked_train_items = self._verify_checked_train_items(input_data)
        examinations = self._load_checked_examinations(checked_train_items[EXAMINATIONS_KEY])
        print(examinations)

    def _verify_checked_train_items(self, input_data):
        examinations = input_data[EXAMINATIONS_KEY]
        examinations = [self.examinations[i] for i in range(len(examinations)) if examinations[i] is not 0]

        features_checked = input_data[FEATURES_KEY]
        features_checked = [self.features_titles[i] for i in range(len(features_checked)) if
                            features_checked[i] is not 0]

        window_width = input_data[WINDOW_WIDTH_KEY]
        return {
            EXAMINATIONS_KEY: examinations,
            FEATURES_KEY: features_checked,
            WINDOW_WIDTH_KEY: window_width
        }

    def _prepare_train_test_data(self, titles):
        for title in titles:
            # load examination
            examination = Examination(title)
            # split by stages

            # calculate feature with window
            # (class, [features])
            # shuffle data set
            # split percentage

    def _prepare_data_set(self, stages, features_callbacks, window_size=1, stages_mode=StagesMode.REM_VS_NO_REM):
        class_nos = self.stages_to_class_nos(stages)
        data_set = []
        for value, class_no in zip(stages.values(), class_nos):
            data_set.append(
                self._prepare_data_set_for_one_stage(
                    value, class_no, features_callbacks, window_size
                )
            )
        return data_set

    def stages_to_class_nos(self, stages, stages_mode=StagesMode.REM_VS_NO_REM):
        keys = stages.keys()
        class_nos = []
        if stages_mode is StagesMode.REM_VS_NO_REM:
            class_nos = [1 if k is 'keys' else 0 for k in keys]
        elif stages_mode is StagesMode.ALL_STAGES:
            class_nos = [i for i in range(len(keys))]
        else:
            pass
        return class_nos

    def _prepare_data_set_for_one_stage(self, stage, class_no, features_callbacks, window_size=1):
        data_set = []
        for samples in stage:
            calculated_features = self._calculate_features(samples, features_callbacks, window_size)
            data_set.append((class_no, calculated_features))
        return data_set

    def _calculate_features(self, samples, features_callbacks, window_size=1):
        if window_size % 2 is 0:
            raise Exception(WINDOW_SIZE_EXCEPTION_MESSAGE)
        pre = int(window_size / 2)
        post = window_size - pre
        samples_amount = len(samples)
        calculated_features = []
        for i in range(samples_amount):
            if i < pre or i > samples_amount - post:
                continue
            window_samples = samples_amount[i - pre:i + post]
            calculated_features.append(
                self._calculate_features_for_one_sample(window_samples, features_callbacks)
            )
        return calculated_features

    def _calculate_features_for_one_sample(self, window_samples, features_callbacks):
        calculated_features = []
        for feature_callback in features_callbacks:
            calculated_features.append(feature_callback(window_samples))
        return calculated_features

    def on_load_model_button_click(self):
        print('load model')

    def on_save_model_button_click(self):
        print('save model')
