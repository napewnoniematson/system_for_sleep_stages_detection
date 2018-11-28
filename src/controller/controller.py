import matplotlib.pyplot as plt
import random
import numpy as np

from src.model.universite_de_mons_data.examination import Examination, HypnogramType
from src.feature.provider import Provider
import src.filter.butterworth as bf
from src.utils.util import *
from src.ai.model import Model
from src.ai.classifier import Classifier


class Controller:
    SUBJECT_NO = 16

    def __init__(self):
        pass

    def run(self):
        import time
        start = time.time()
        f_train = []
        l_train = []
        f_test = []
        l_test = []
        for i in range(20):
            print("START: ", i)
            subject_nr = i + 1
            self.examination = Examination(subject_nr)
            self.fpi_a2 = self.examination.eeg.load_fpi_a2()

            fs = self.examination.eeg.load_eeg_frequency()
            low_cut = 0.5
            high_cut = 30
            filtered = bf.butter_bandpass_filter(self.fpi_a2, low_cut, high_cut, fs, order=6)

            wake = self.examination.hypnogram['wake']
            rem = self.examination.hypnogram['rem']
            s1 = self.examination.hypnogram['s1']
            s2 = self.examination.hypnogram['s2']
            s3 = self.examination.hypnogram['s3']

            length = len(wake) + len(rem) + len(s1) + len(s2) + len(s3)
            fp = Provider(filtered, length)
            fp.update(wake, 0)
            fp.update(rem, 1)
            fp.update(s1, 2)
            fp.update(s2, 3)
            fp.update(s3, 4)

            data_set = fp.get_data_set()
            random.shuffle(data_set)
            random.shuffle(data_set)
            random.shuffle(data_set)
            split_index = int(length * 0.8)
            training = data_set[:split_index]
            test = data_set[split_index:]

            for sample in training:
                f_train.append(sample[:4])
                l_train.append(sample[4])

            for sample in test:
                f_test.append(sample[:4])
                l_test.append(sample[4])

        f_train = np.array(f_train)
        l_train = np.array(l_train)
        f_test = np.array(f_test)
        l_test = np.array(l_test)

        model = Model(f_train, l_train, has_normalization=True)
        classifier = Classifier(model.model)
        res = classifier.evaluate(f_test, l_test)
        print("***RESULT***")
        print(res)
        wyn = classifier.verify(f_test)
        print("wyn: {} | prawdziwy: {}".format(wyn, l_test[0]))
        stop = time.time()
        print("***TIME***\n", stop - start)


if __name__ == '__main__':
    Controller().run()
