from src.utils.util import *


class TrainController:
    def __init__(self):
        self.examinations = ["subject{}".format(i) for i in range(20)]
        self.features = ["min", "max", "variety", "value", "mean", "rms"]
        self.window_sizes = [("{}s".format((i + 1) * 5), (i + 1) * 5) for i in range(5)]

    def on_train_button_click(self, input_data):
        examinations = input_data[EXAMINATIONS_KEY]
        examinations = [i for i in range(len(examinations)) if examinations[i] is not 0]

        features = input_data[FEATURES_KEY]
        features = [i for i in range(len(features)) if features[i] is not 0]

        window_width = input_data[WINDOW_WIDTH_KEY]

        print(examinations, features, window_width)

    def on_load_model_button_click(self):
        print('load model')

    def on_save_model_button_click(self):
        print('save model')
