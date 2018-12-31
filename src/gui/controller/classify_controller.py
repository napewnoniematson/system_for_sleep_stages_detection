from src.utils.util import *
import src.data_set.full_set as ds
import src.data_set.train_test_set as tts


class ClassifyController:
    def __init__(self, result=None, classifier_callback=None):
        # todo when sub_19 is clicked it use sub_20, probably radiobutton returns values from range 1 to 20 not 0 to 19
        # todo so need to decrease read value by 1
        # todo check in train_controller
        self.examinations = [("subject{}".format(i + 1), i + 1) for i in range(20)]
        self.result = result
        self.classifier = None
        self.classifier_callback = classifier_callback

    # todo classify
    def on_classify_button_click(self, input_data):
        print(input_data)
        import time
        start = time.time()
        checked_items = self._verify_checked_classify_items(input_data)
        classify_examination = checked_items[CLASSIFY_KEY]

        import src.feature.provider as p
        self.result.features = self.result.features if self.result.features else [p.root_mean_square]
        self.result.window_width = self.result.window_width if self.result.window_width else 5
        self.result.stage_mode = self.result.stage_mode if self.result.stage_mode else 0

        data_set = ds.prepare_data_set(
            classify_examination,
            self.result.features,
            self.result.window_width,
            self.result.stage_mode
            # todo **kwargs for features callback functions
        )
        print(data_set)
        data_set = ds.flatten_data_set(data_set)
        print(data_set)
        features = tts.split_data_set_numpy(data_set)[1]
        stop = time.time()
        data_time = stop - start
        start = time.time()
        self.classifier = self.classifier_callback()
        import src.loader.aasm as aasm
        path = UNIVERSITE_DE_MONS_HYPNOGRAM_AASM.format(classify_examination[0])
        print(path)
        hypno = aasm.load(path)
        temp = []
        a = len(hypno['data'])
        val2 = 0
        for i in range(a):
            if '1' == hypno['data'][i]:
                val2 == 1
                # print(i)
            else:
                val2 == 0
            for _ in range(1000):
                temp.append(val2)

        if self.classifier:
            results = [self.classifier.verify(fs) for fs in features[1800000:2000000]]
            counter = 0
            for i in range(2000000-1800000):
                if results[i] == temp[1800000 + i]:
                    counter = counter + 1
            print(counter)
        else:
            print("Classifier won't work")
        stop = time.time()
        classifier_time = stop - start
        print("Data_time: {}\nClassifier_time: {}".format(data_time, classifier_time))
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
