import datetime

from src.utils.util import *
import src.feature.provider as features
from src.model.universite_de_mons_data.examination import Examination, HypnogramType
from src.ai.model import Model
from src.ai.classifier import Classifier
from src.model.stage_mode import StageMode


class TrainController:
    def __init__(self, result=None):
        self.features_callbacks = features.callbacks()
        self.features_titles = [c.__name__ for c in self.features_callbacks]
        self.examinations = ["subject{}".format(i + 1) for i in range(20)]
        self.window_sizes = [("{}s".format(i * 10 + 5), i * 10 + 5) for i in range(5)]
        self.stage_modes = [('REM vs NREM', StageMode.REM_VS_NREM.value), ('All stages', StageMode.ALL_STAGES.value)]
        self.epochs = [(str(i), i) for i in [1, 2, 5, 10, 15]]
        self.training_part_percentage = [('{}%'.format((i + 1) * 12.5), (i + 1) * 12.5) for i in range(7)]
        self.model = None
        self.classifier = None
        self.model_name = None
        self.result = result

    def on_train_button_click(self, input_data):
        checked_train_items = self._verify_checked_train_items(input_data)
        data_set = self._prepare_data_set(
            checked_train_items[EXAMINATIONS_KEY],
            checked_train_items[FEATURES_KEY],
            checked_train_items[WINDOW_WIDTH_KEY],
            checked_train_items[STAGE_MODE_KEY]
            # todo **kwargs for features callback functions
        )
        data_set = self._flatten_data_set(data_set)
        l_train, f_train, l_test, f_test = self._prepare_train_test_sets(data_set,
                                                                         training_part_percentage=checked_train_items[
                                                                             TRAINING_PART_KEY])
        # self.model = self._prepare_model(f_train, l_train, epochs=checked_train_items[EPOCHS_KEY])
        # self.classifier = self._prepare_classifier(self.model)
        # accuracy = self._evaluate_classifier_efficiency(self.classifier, f_test, l_test)[1]
        self.update_result(
            checked_train_items[EXAMINATIONS_KEY],
            [f.__name__ for f in checked_train_items[FEATURES_KEY]],
            checked_train_items[WINDOW_WIDTH_KEY],
            StageMode(checked_train_items[STAGE_MODE_KEY]).name,
            checked_train_items[EPOCHS_KEY],
            checked_train_items[TRAINING_PART_KEY],
            "name",
            0
        )

    def update_result(self, examinations, features, window_width, stage_mode, epochs,
                      training_part, model_name, accuracy):
        if self.result:
            self.result.examinations = examinations
            self.result.features = features
            self.result.window_width = window_width
            self.result.stage_mode = stage_mode
            self.result.epochs = epochs
            self.result.training_part = training_part
            self.result.model_name = model_name
            self.result.accuracy = accuracy

    def _verify_checked_train_items(self, input_data):
        examinations = input_data[EXAMINATIONS_KEY]
        examinations = [self.examinations[i] for i in range(len(examinations)) if examinations[i] is not 0]

        features_checked = input_data[FEATURES_KEY]
        features_checked = [self.features_callbacks[i] for i in range(len(features_checked)) if
                            features_checked[i] is not 0]

        window_width = input_data[WINDOW_WIDTH_KEY]

        stage_mode = input_data[STAGE_MODE_KEY]

        epochs = input_data[EPOCHS_KEY]

        training_part = input_data[TRAINING_PART_KEY]

        return {
            EXAMINATIONS_KEY: examinations,
            FEATURES_KEY: features_checked,
            WINDOW_WIDTH_KEY: window_width,
            STAGE_MODE_KEY: stage_mode,
            EPOCHS_KEY: epochs,
            TRAINING_PART_KEY: training_part
        }

    def _prepare_model(self, f_train, l_train, has_normalization=True, epochs=EPOCHS_DEFAULT):
        return Model(f_train, l_train, has_normalization, epochs)

    def _prepare_classifier(self, model):
        return Classifier(model.model)

    def _evaluate_classifier_efficiency(self, classifier, f_test, l_test):
        return classifier.evaluate(f_test, l_test)

    def _prepare_train_test_sets(self, data_set, training_part_percentage=TRAINING_PART_PERCENTAGE_DEFAULT):
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

    def _prepare_data_set(self, examination_titles, features_callbacks, window_width,
                          stage_mode, **kwargs):
        data_set = []
        for title in examination_titles:
            # load examination
            examination = Examination(title)
            eeg = examination.eeg.load_fpi_a2()
            stages_indices = self._get_stages_indices(examination)
            stages = self._convert_indices_to_values(eeg, stages_indices)
            data_set.append(
                self._prepare_data_set_for_one_examination(
                    stages, features_callbacks, window_width, stage_mode, **kwargs
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
    def _prepare_data_set_for_one_examination(self, stages, features_callbacks, window_width=WINDOW_WIDTH_DEFAULT,
                                              stage_mode=STAGE_MODE_VALUE_DEFAULT, **kwargs):
        class_nos = self.stages_to_class_nos(stages, stage_mode)
        data_set = []
        for value, class_no in zip(stages.values(), class_nos):
            data_set.append(
                self._prepare_data_set_for_one_stage(
                    value, class_no, features_callbacks, window_width, **kwargs
                )
            )
        return data_set

    def stages_to_class_nos(self, stages, stage_mode=STAGE_MODE_VALUE_DEFAULT):
        keys = stages.keys()
        class_nos = []
        if StageMode.REM_VS_NREM.value == stage_mode:
            class_nos = [1 if k is 'rem' else 0 for k in keys]
        elif StageMode.ALL_STAGES.value is stage_mode:
            class_nos = [i for i in range(len(keys))]
        else:
            pass
        return class_nos

    def _prepare_data_set_for_one_stage(self, stage, class_no, features_callbacks,
                                        window_width=WINDOW_WIDTH_DEFAULT, **kwargs):
        calculated_features_set = []
        for samples in stage:
            calculated_features = self._calculate_features(samples, features_callbacks, window_width, **kwargs)
            calculated_features_set.append(calculated_features)
        data_set = [[(class_no, cf) for cf in calculated_features] for calculated_features in calculated_features_set]
        return data_set

    def _calculate_features(self, samples, features_callbacks, window_width=WINDOW_WIDTH_DEFAULT, **kwargs):
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
        self.model_name, self.model = Model.loads()
        self.classifier = Classifier(self.model.model)
        self.update_result(None, None, None, None, self.model_name, None)

    def on_save_model_button_click(self):
        if self.model:
            date = self.get_time()
            self.model_name = MODEL_NAME.format(date) + MODEL_FILE_EXTENSION
            self.model.save(file=self.model_name)

    def get_time(self):
        return datetime.datetime.now().strftime(LOG_DATE_TIME_FORMAT)
