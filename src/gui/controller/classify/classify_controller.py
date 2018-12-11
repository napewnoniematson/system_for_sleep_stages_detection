from src.utils.util import *
import src.data_set.full_set as ds
import src.data_set.train_test_set as tts

class ClassifyController:
    def __init__(self, result=None, classifier=None):
        self.examinations = [("subject{}".format(i+1), i+1) for i in range(20)]
        self.result = result
        self.classifier = classifier

    # todo classify
    def on_classify_button_click(self, input_data):
        checked_items = self._verify_checked_classify_items(input_data)
        classify_examination = checked_items[CLASSIFY_KEY]
        data_set = ds.prepare_data_set(
            classify_examination,
            self.result.features,
            self.result.window_width,
            self.result.stage_mode
            # todo **kwargs for features callback functions
        )
        data_set = ds.flatten_data_set(data_set)
        features = tts.split_data_set_numpy(data_set)[1]
        if self.classifier:
            results = [self.classifier.verify(fs) for fs in features]
            print(results)
        self.update_result(
            classify_examination,
            len(features)
        )

    def _verify_checked_classify_items(self, input_data):
        examination = [self.examinations[int(input_data)][0]]
        return {
            CLASSIFY_KEY: examination
        }

    def update_result(self, classify_examination, classify_examination_length):
        if self.result:
            self.result.classify_examination = classify_examination
            self.result.classify_examination_length = classify_examination_length
