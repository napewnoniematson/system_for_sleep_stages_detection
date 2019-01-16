from src.decorator.timer import timer
import src.utils.timer as u_timer
import src.loader.physionet_sleep_cassette.titles as titles
import src.data_set.input_set as input_data
import src.data_set.train_test_set as train_data
import src.feature_pack.feature as features
import src.ai.ann.model as ann_m
import src.ai.ann.classifier as ann_c
import src.ai.k_nn.classifier as knn_c
import src.ai.k_nn2.classifier as knn2_c
from src.utils.util import *
import src.data_set.data as data
from src.file_system.directory_manager import *
import time
import numpy as np

physio_hypnogram, physio_signals = titles.get(PHYSIONET_DIR)
create_main_directory()


@timer
def case_1():
    """
    TRAINING/TEST
        75/25
    WINDOW:
        1x: 5s
    FEATURES:
        EOG: energy, median, max
        EEG: energy of rhythm (alpha, beta, theta, delta), energy ratio
    RESULT:
        AWAKE/REM/NREM
    ANN:
        input(11), 32, 18, 2
    EPOCHS:
        5
    """

    f = [
        [
            features.alpha_fft_energy, features.alpha_fft_energy_ratio,
            features.beta_fft_energy, features.beta_fft_energy_ratio,
            features.theta_fft_energy, features.theta_fft_energy_ratio,
            features.delta_fft_energy, features.delta_fft_energy_ratio
        ],
        [
            features.energy, features.median, features.max_peak
        ]
    ]
    window = 5 * 200
    checked = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]

    data_set = input_data.examinations_data_set(udm_exams.copy(), f, window)
    l_train, f_train, l_test, f_test = train_data.train_test_data(data_set, checked)
    model = ann_m.Model(f_train, l_train, hidden1=32, hidden2=18, has_normalization=False, epochs=5)
    classifier = ann_c.Classifier(model)
    evaluate = classifier.evaluate_em(f_test, l_test)
    print(evaluate)


@timer
def case_2():
    """
    TRAINING/TEST
        75/25
    WINDOW:
        1x: 5s
    FEATURES:
        EOG: max
        EEG: rms, median, max
    RESULT:
        AWAKE/REM/NREM
    KNN:
        k=7
    """

    f = [
        [
            features.root_mean_square, features.median, features.max_peak
        ],
        [
            features.max_peak
        ]
    ]
    window = 5 * 200
    # checked = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
    checked = [-1, -1, -1, -1, -1, -1, -1, -1, 1, -1, -1, -1, 0, 0, 1, 1, 1, 1, 1, 1]

    data_set = input_data.examinations_data_set(udm_exams.copy(), f, window)
    l_train, f_train, l_test, f_test = train_data.train_test_data(data_set, checked)
    classifier = knn_c.Classifier(f_train, l_train, k=1)
    evaluate = classifier.evaluate(f_test, l_test)
    print(evaluate)


@timer
def case_3():
    """
    test
    """
    f = [
        [
            features.alpha_fft_energy, features.alpha_fft_energy_ratio,
            features.beta_fft_energy, features.beta_fft_energy_ratio,
            features.theta_fft_energy, features.theta_fft_energy_ratio,
            features.delta_fft_energy, features.delta_fft_energy_ratio
        ],
        [
            features.energy, features.median, features.max_peak
        ]
    ]
    window = 5 * 200
    checked = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
    data_set = input_data.examinations_data_set(udm_exams.copy(), f, window)
    l_train, f_train, l_test, f_test = train_data.train_test_data(data_set, checked)
    model = ann_m.Model(f_train, l_train, hidden1=32, hidden2=18, has_normalization=False, epochs=5)
    classifier = ann_c.Classifier(model)
    evaluate = classifier.evaluate(f_test, l_test)
    print(evaluate)


@timer
def case_4():
    """
    test
    """
    f = [
        [
            features.alpha_fft_energy, features.alpha_fft_energy_ratio,
            features.beta_fft_energy, features.beta_fft_energy_ratio,
            features.theta_fft_energy, features.theta_fft_energy_ratio,
            features.delta_fft_energy, features.delta_fft_energy_ratio
        ],
        [
            features.energy, features.median, features.max_peak
        ]
    ]
    window = 5 * 200
    checked = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
    data_set = input_data.examinations_data_set(udm_exams.copy(), f, window)
    l_train, f_train, l_test, f_test = train_data.train_test_data(data_set, checked)
    classifier = knn2_c.Classifier(f_train, l_train, 5)
    evaluate = classifier.evaluate(f_test, l_test)
    print(evaluate)


@timer
def case_5():
    """
        test
        """
    f = [
        [
            features.alpha_fft_energy, features.alpha_fft_energy_ratio,
            features.beta_fft_energy, features.beta_fft_energy_ratio,
            features.theta_fft_energy, features.theta_fft_energy_ratio,
            features.delta_fft_energy, features.delta_fft_energy_ratio
        ],
        [
            features.energy, features.median, features.max_peak
        ]
    ]
    window = 5 * 200
    # checked = [1]
    data_set = input_data.examinations_data_set(udm_exams.copy(), f, window)
    l_train, f_train, l_test, f_test = train_data.train_test_data_one_examination(data_set[0])
    classifier = knn2_c.Classifier(f_train, l_train, 5)
    evaluate = classifier.evaluate(f_test, l_test)
    print(evaluate)


@timer
def case_6():
    """
    test
    """
    f = [
        [
            features.alpha_fft_energy, features.alpha_fft_energy_ratio,
            features.beta_fft_energy, features.beta_fft_energy_ratio,
            features.theta_fft_energy, features.theta_fft_energy_ratio,
            features.delta_fft_energy, features.delta_fft_energy_ratio
        ],
        [
            features.energy, features.median, features.max_peak
        ]
    ]
    window = 5 * 200
    # checked = [1]
    data_set = input_data.examinations_data_set(udm_exams.copy(), f, window)
    l_train, f_train, l_test, f_test = train_data.train_test_data_one_examination(data_set[0])
    model = ann_m.Model(f_train, l_train, hidden1=32, hidden2=18, has_normalization=False, epochs=5)
    classifier = ann_c.Classifier(model)
    evaluate = classifier.evaluate(f_test, l_test)
    print(evaluate)


@timer
def c7():
    """
        test
        """
    print("c7 started")
    f = [
        [
            features.median, features.harmonic_mean, features.mean
        ],
        [
            features.harmonic_mean,
        ]
    ]
    window = 5 * 100
    checked = [0 if i % 4 == 0 else 1 for i in range(len(physio_exams))]
    data_set = input_data.examinations_data_set(physio_exams.copy(), f, window)
    l_train, f_train, l_test, f_test = train_data.train_test_data(data_set, checked)
    model = ann_m.Model(f_train, l_train, hidden1=32, hidden2=18, has_normalization=False, epochs=5)
    classifier = ann_c.Classifier(model)
    evaluate = classifier.evaluate(f_test, l_test)
    print(evaluate)


@timer
def c8():
    registered_time = u_timer.get_time()
    create_data_directory(registered_time)
    create_training_directory(registered_time)
    create_test_directory(registered_time)
    f = [[features.mean, features.median], [features.max_peak]]
    window = 5 * 100
    print("TRAINING: ")
    l = 0
    for i in f:
        l += len(i)
    print(l)
    model = ann_m.Model(input_amount=l, output_amount=3, hidden1=32, hidden2=18, has_normalization=False, epochs=5)
    cls = ann_c.Classifier(model)
    for data_set in data.examinations_data_set(
            physio_signals[107:117], physio_hypnogram[107:117], f, window, True, registered_time
    ):
        f_train, l_train = data.split(data_set)
        model.train(f_train, l_train, 5)

    print("TEST: ")
    for data_set in data.examinations_data_set(
            physio_signals[65:67], physio_hypnogram[65:67], f, window, False, registered_time
    ):
        f_test, l_test = data.split(data_set)
        evaluate = cls.evaluate2(f_test, l_test)
        print(evaluate)

