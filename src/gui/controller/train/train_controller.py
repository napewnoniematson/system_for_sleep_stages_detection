from enum import Enum

from src.utils.util import *
import src.feature.provider as features
from src.model.universite_de_mons_data.examination import Examination, HypnogramType
from src.ai.model import Model
from src.ai.classifier import Classifier


class StagesMode(Enum):
    REM_VS_NO_REM = 0,
    ALL_STAGES = 1


class TrainController:
    def __init__(self):
        self.features_callbacks = features.callbacks()
        self.features_titles = [c.__name__ for c in self.features_callbacks]
        self.examinations = ["subject{}".format(i) for i in range(20)]
        self.window_sizes = [("{}s".format(i * 10 + 5), i * 10 + 5) for i in range(5)]
        self.model = None
        self.classifier = None

    def on_train_button_click(self, input_data):
        checked_train_items = self._verify_checked_train_items(input_data)
        data_set = self._prepare_data_set(
            checked_train_items[EXAMINATIONS_KEY],
            checked_train_items[FEATURES_KEY],
            checked_train_items[WINDOW_WIDTH_KEY]
            # todo **kwargs for features callback functions
        )
        data_set = self._flatten_data_set(data_set)
        l_train, f_train, l_test, f_test = self._prepare_train_test_sets(data_set)
        # todo epochs from radiobox
        self.model = self._prepare_model(f_train, l_train, epochs=2)
        self.classifier = self._prepare_classifier(self.model)
        print(self._evaluate_classifier_efficiency(
            self.classifier, f_test, l_test
        ))

    def _verify_checked_train_items(self, input_data):
        examinations = input_data[EXAMINATIONS_KEY]
        examinations = [self.examinations[i] for i in range(len(examinations)) if examinations[i] is not 0]

        features_checked = input_data[FEATURES_KEY]
        features_checked = [self.features_callbacks[i] for i in range(len(features_checked)) if
                            features_checked[i] is not 0]

        window_width = input_data[WINDOW_WIDTH_KEY]
        return {
            EXAMINATIONS_KEY: examinations,
            FEATURES_KEY: features_checked,
            WINDOW_WIDTH_KEY: window_width
        }

    def _prepare_model(self, f_train, l_train, has_normalization=True, epochs=15):
        return Model(f_train, l_train, has_normalization, epochs)

    def _prepare_classifier(self, model):
        return Classifier(model.model)

    def _evaluate_classifier_efficiency(self, classifier, f_test, l_test):
        return classifier.evaluate(f_test, l_test)

    def _prepare_train_test_sets(self, data_set, training_part_percentage=70):
        self._extreme_shuffle_data_set(data_set)
        split_index = int(len(data_set) * training_part_percentage / 100)
        training = data_set[:split_index]
        test = data_set[split_index:]
        l_train, f_train = self._split_data_set_numpy(training)
        l_test, f_test = self._split_data_set_numpy(test)
        return l_train, f_train, l_test, f_test

    def _split_data_set_numpy(self, data_set):
        import numpy as np
        l_data, f_data = self._split_data_set(data_set)
        return np.array(l_data), np.array(f_data)

    def _split_data_set(self, data_set):
        l_data = []
        f_data = []
        for data in data_set:
            l_data.append(data[0])
            f_data.append(data[1])
        return l_data, f_data

    def _prepare_data_set(self, examination_titles, features_callbacks, window_width, **kwargs):
        data_set = []
        for title in examination_titles:
            # load examination
            examination = Examination(title)
            eeg = examination.eeg.load_fpi_a2()
            stages_indices = self._get_stages_indices(examination)
            stages = self._convert_indices_to_values(eeg, stages_indices)
            data_set.append(
                self._prepare_data_set_for_one_examination(
                    stages, features_callbacks, window_width, **kwargs
                )
            )
        return data_set

    def _extreme_shuffle_data_set(self, data_set):
        import random
        for _ in range(3):
            random.shuffle(data_set)

    def _flatten_data_set(self, data_set):
        temp = []
        for examination in data_set:
            for i in examination:
                for j in i:
                    for x in j:
                        temp.append(x)
        return temp

    def _convert_indices_to_values(self, eeg, stages_indices):
        return {key: [eeg[indices[0]:indices[1]] for indices in value] for key, value in stages_indices.items()}

    def _get_stages_indices(self, examination):
        return {key: value for key, value in examination.hypnogram.items()
                if not (key is 'title' or key is 'data')}

    # todo stages_mode from radiobox
    def _prepare_data_set_for_one_examination(self, stages, features_callbacks, window_width=1,
                                              stages_mode=StagesMode.REM_VS_NO_REM, **kwargs):
        class_nos = self.stages_to_class_nos(stages, stages_mode)
        data_set = []
        for value, class_no in zip(stages.values(), class_nos):
            data_set.append(
                self._prepare_data_set_for_one_stage(
                    value, class_no, features_callbacks, window_width, **kwargs
                )
            )
        return data_set

    def stages_to_class_nos(self, stages, stages_mode=StagesMode.REM_VS_NO_REM):
        keys = stages.keys()
        class_nos = []
        if stages_mode is StagesMode.REM_VS_NO_REM:
            class_nos = [1 if k is 'rem' else 0 for k in keys]
        elif stages_mode is StagesMode.ALL_STAGES:
            class_nos = [i for i in range(len(keys))]
        else:
            pass
        return class_nos

    def _prepare_data_set_for_one_stage(self, stage, class_no, features_callbacks, window_width=1, **kwargs):
        calculated_features_set = []
        for samples in stage:
            calculated_features = self._calculate_features(samples, features_callbacks, window_width, **kwargs)
            calculated_features_set.append(calculated_features)
        data_set = [[(class_no, cf) for cf in calculated_features] for calculated_features in calculated_features_set]
        return data_set

    def _calculate_features(self, samples, features_callbacks, window_width=1, **kwargs):
        if window_width % 2 is 0:
            raise Exception(WINDOW_SIZE_EXCEPTION_MESSAGE)
        pre = int(window_width / 2)
        post = window_width - pre
        samples_amount = len(samples)
        calculated_features = []
        for i in range(samples_amount):
            if i < pre or i > samples_amount - post:
                continue
            window_samples = samples[i - pre:i + post]
            calculated_features.append(
                self._calculate_features_for_one_sample(window_samples, features_callbacks, **kwargs)
            )
        return calculated_features

    def _calculate_features_for_one_sample(self, window_samples, features_callbacks, **kwargs):
        calculated_features = []
        for feature_callback in features_callbacks:
            calculated_features.append(feature_callback(window_samples, **kwargs))
        return calculated_features

    def on_load_model_button_click(self):
        self.model = Model.loads()
        self.classifier = Classifier(self.model.model)

    def on_save_model_button_click(self):
        if self.model:
            self.model.save()
