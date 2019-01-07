import time

start = time.time()
subject_no = 20  # 20
examinations_titles = ["subject{}".format(i + 1) for i in range(subject_no)]
print(examinations_titles)

import src.data_set.full_set as fs

ex = fs.load_examinations(examinations_titles)
from src.feature_pack.feature import *

data_set = fs.examinations_data_set(
    ex,
    [
        [
            alpha_energy, alpha_energy_ratio,
            beta_energy, beta_energy_ratio,
            theta_energy, theta_energy_ratio,
            delta_energy, delta_energy_ratio
        ],
        [
            value, mean, max_peak
        ]
    ],
    1000,
)
import src.data_set.train_test_set as tss

l_train, f_train, l_test, f_test = tss.train_test_data(data_set)

from src.ai.ann.model import Model

model = Model(f_train, l_train, has_normalization=True, epochs=5)

from src.ai.ann.classifier import Classifier

classifier = Classifier(model)
evaluate = classifier.evaluate(f_test, l_test)
print(evaluate)
stop = time.time()
print(stop - start)
