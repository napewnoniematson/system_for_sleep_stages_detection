from src.utils.util import *
import src.feature.provider as features
from src.model.universite_de_mons_data.examination import Examination, HypnogramType

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

    # TODO one big loop or many small methods
    def _load_checked_examinations(self, titles):
        examinations = []
        for title in titles:
            examination = Examination(title)
            eeg = examination.eeg.load_fpi_a2()
            fs = examination.eeg.load_eeg_frequency()
            examination = (title, examination)
            examinations.append(examination)
        return examinations

    def on_load_model_button_click(self):
        print('load model')

    def on_save_model_button_click(self):
        print('save model')
