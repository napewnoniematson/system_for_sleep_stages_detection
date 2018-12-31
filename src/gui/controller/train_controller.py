import datetime

from src.utils.util import *
import src.feature.provider as features
# from src.model.universite_de_mons_data.examination import Examination
from src.ai.model import Model
from src.ai.classifier import Classifier
from src.model.stage_mode import StageMode
import src.data_set.full_set as ds
import src.data_set.train_test_set as tss


class TrainController:
    def __init__(self, result=None):
        self.features_callbacks = features.callbacks()
        self.features_titles = [c.__name__ for c in self.features_callbacks]
        self.examinations_titles = ["subject{}".format(i + 1) for i in range(20)]
        self.window_sizes = [("{}sec".format((i + 1)), (i + 1) * 200) for i in range(10)]
        self.stage_modes = [('REM vs NREM', StageMode.REM_VS_NREM.value), ('All stages', StageMode.ALL_STAGES.value)]
        self.epochs = [(str(i), i) for i in [1, 2, 5, 10, 15]]
        self.model = None
        self.classifier = None
        self.model_name = None
        self.result = result
        self.examinations = ds.load_examinations(self.examinations_titles)

    def on_train_button_click(self, input_data):
        checked_train_items = self._verify_checked_train_items(input_data)
        print(input_data)

        data_set = ds.examinations_data_set(
            self.examinations,
            checked_train_items[FEATURES_KEY],
            checked_train_items[WINDOW_WIDTH_KEY],
            checked_train_items[STAGE_MODE_KEY]
            # todo **kwargs for features callback functions
        )
        l_train, f_train, l_test, f_test = tss.train_test_data(data_set, checked_train_items[EXAMINATIONS_KEY])
        self.model = Model(f_train, l_train, has_normalization=True, epochs=int(checked_train_items[EPOCHS_KEY]),
                           stage_mode=checked_train_items[STAGE_MODE_KEY])
        self.model_name = MODEL_NAME.format(self.get_time()) + MODEL_FILE_EXTENSION
        self.classifier = Classifier(self.model)
        evaluate = self.classifier.evaluate(f_test, l_test)
        print(evaluate)
        self.update_result(
            [self.examinations_titles[i] for i in range(len(checked_train_items[EXAMINATIONS_KEY])) if
             checked_train_items[EXAMINATIONS_KEY][i] is not 0],
            checked_train_items[FEATURES_KEY],
            checked_train_items[WINDOW_WIDTH_KEY],
            checked_train_items[STAGE_MODE_KEY],
            checked_train_items[EPOCHS_KEY],
            self.model_name,
            evaluate['accuracy']
        )

    def update_result(self, examinations, features, window_width, stage_mode, epochs, model_name, accuracy):
        if self.result:
            self.result.examinations = examinations
            self.result.features = features
            self.result.window_width = window_width
            self.result.stage_mode = stage_mode
            self.result.epochs = epochs
            self.result.model_name = model_name
            self.result.accuracy = accuracy

    def _verify_checked_train_items(self, input_data):
        examinations = input_data[EXAMINATIONS_KEY]

        features_checked = input_data[FEATURES_KEY]
        features_checked = [self.features_callbacks[i] for i in range(len(features_checked)) if
                            features_checked[i] is not 0]

        window_width = int(input_data[WINDOW_WIDTH_KEY])

        stage_mode = int(input_data[STAGE_MODE_KEY])

        epochs = int(input_data[EPOCHS_KEY])

        return {
            EXAMINATIONS_KEY: examinations,
            FEATURES_KEY: features_checked,
            WINDOW_WIDTH_KEY: window_width,
            STAGE_MODE_KEY: stage_mode,
            EPOCHS_KEY: epochs
        }

    def on_load_model_button_click(self):
        self.model_name, self.model = Model.loads(file='model_2018-12-12 20:23:31.model')
        self.classifier = Classifier(self.model)
        print(self.classifier)
        self.update_result(None, None, None, None, None, None, self.model_name, None)

    def on_save_model_button_click(self):
        if self.model:
            self.model.save(file=self.model_name)

    def get_classifier(self):
        return self.classifier

    @staticmethod
    def get_time():
        return datetime.datetime.now().strftime(LOG_DATE_TIME_FORMAT)
