from src.decorator.timer import timer
import src.data_set.input_set as input_data
import src.data_set.train_test_set as train_data
import src.feature_pack.feature as features
import src.ai.ann.model as ann_m
import src.ai.ann.classifier as ann_c
import src.ai.k_nn.classifier as knn_c

titles = ["subject{}".format(i + 1) for i in range(20)]
exams = input_data.load_examinations(titles)


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
            features.alpha_energy, features.alpha_energy_ratio,
            features.beta_energy, features.beta_energy_ratio,
            features.theta_energy, features.theta_energy_ratio,
            features.delta_energy, features.delta_energy_ratio
        ],
        [
            features.energy, features.median, features.max_peak
        ]
    ]
    window = 5 * 200
    checked = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]

    data_set = input_data.examinations_data_set(exams.copy(), f, window)
    l_train, f_train, l_test, f_test = train_data.train_test_data(data_set, checked)
    model = ann_m.Model(f_train, l_train, hidden1=32, hidden2=18, has_normalization=False, epochs=5)
    classifier = ann_c.Classifier(model)
    evaluate = classifier.evaluate(f_test, l_test)
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
    checked = [-1, -1, -1, -1, -1, -1, -1, -1, 1, -1, -1, -1, -1, 0, -1, -1, 1, 1, 1, -1]

    data_set = input_data.examinations_data_set(exams.copy(), f, window)
    l_train, f_train, l_test, f_test = train_data.train_test_data(data_set, checked)
    classifier = knn_c.Classifier(f_train, l_train, k=7)
    evaluate = classifier.evaluate(f_test, l_test)
    print(evaluate)