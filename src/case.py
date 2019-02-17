from src.decorator.timer import timer
import src.utils.timer as u_timer
import src.loader.physionet_sleep_cassette.titles as titles
import src.feature_pack.feature as features
import src.ai.ann.model as ann_m
import src.ai.ann.classifier as ann_c
import src.data_set.data as data
from src.file_system.directory_manager import *

physio_hypnogram, physio_signals = titles.get(PHYSIONET_DIR)
training_hypnogram, training_signals, test_hypnogram, test_signals = [], [], [], []
for i in range(len(physio_signals)):
    if i % 16 == 0:
        test_hypnogram.append(physio_hypnogram[i])
        test_signals.append(physio_signals[i])
    else:
        training_hypnogram.append(physio_hypnogram[i])
        training_signals.append(physio_signals[i])
print(len(training_signals))
print(len(test_signals))
create_main_directory()
create_model_directory()

default_model_config = {
    'input_amount': 256,
    'output_amount': 3,
    'hidden1': 32,
    'hidden': 18,
    'has_normalization': False,
    # activation1
    # activation2
    # activation3
    # optimizer etc
    'epochs': 5,
    'batch_size': 256
}


def __case(features_callbacks, window, model_config=default_model_config):
    # start time
    registered_time = u_timer.get_time()
    # create directories
    create_data_directory(registered_time)
    create_training_directory(registered_time)
    create_test_directory(registered_time)
    # amount of features
    input_amount = 0
    for callback in features_callbacks:
        input_amount += len(callback)
    # prepare model
    model = ann_m.Model(input_amount=input_amount, output_amount=3, hidden1=32, hidden2=18)
    # prepare classifier
    cls = ann_c.Classifier(model)
    # training part - calculate features + train model
    train_counter = 0
    for data_set in data.examinations_data_set(
            training_signals, training_hypnogram, features_callbacks, window, True, registered_time
    ):
        f_train, l_train = data.split(data_set)
        model.train(f_train, l_train, epochs=7, batch_size=256)
        train_counter += 1
        print("FINISHED {}/{}".format(train_counter, len(training_signals)))
    # testing part - calculate features + evaluate model
    evaluates = []
    for data_set in data.examinations_data_set(
            test_signals, test_hypnogram, features_callbacks, window, False, registered_time
    ):
        f_test, l_test = data.split(data_set)
        evaluate = cls.evaluate2(f_test, l_test)
        print(evaluate)
        evaluates.append(evaluate)
        print("FINISHED {}/{}".format(len(evaluates), len(test_signals)))
    # save model
    model.save(registered_time)
    return {
        'features_callbacks': features_callbacks,
        'window': window,
        'model_config': model_config,
        'trained': train_counter,
        'tested': len(evaluates),
        'evaluates': evaluates
    }


@timer
def case_2():
    features_callbacks = [
        [features.harmonic_mean, features.variance],
        [features.median, features.mean]
    ]
    window = 100
    return __case(features_callbacks=features_callbacks, window=window)
    # 0.778377358490566
    # FINISHED 1/10
    # 0.7516452074391988
    # FINISHED 2/10
    # 0.6901818181818182
    # FINISHED 3/10
    # 0.7310824108241082
    # FINISHED 4/10
    # 0.6681065239551478
    # FINISHED 5/10
    # 0.801410998552822
    # FINISHED 6/10
    # 0.6492363636363636
    # FINISHED 7/10
    # 0.628106808402936
    # FINISHED 8/10
    # 0.658096926713948
    # FINISHED 9/10
    # 0.654457801152016
    # FINISHED 10/10
    # INFO: Method: case_2 finished with time 14603.398421287537 seconds
    # {'features_callbacks': [[<function harmonic_mean at 0x7f2a8fc4db70>, <function variance at 0x7f2a8fc4dbf8>], [<function median at 0x7f2a8fc4dd90>, <function mean at 0x7f2a8fc4dae8>]], 'window': 100, 'model_config': {'input_amount': 256, 'output_amount': 3, 'hidden1': 32, 'hidden': 18, 'has_normalization': False, 'epochs': 5, 'batch_size': 256}, 'trained': 135, 'tested': 10, 'evaluates': [0.778377358490566, 0.7516452074391988, 0.6901818181818182, 0.7310824108241082, 0.6681065239551478, 0.801410998552822, 0.6492363636363636, 0.628106808402936, 0.658096926713948, 0.654457801152016]}


@timer
def case_3():
    features_callbacks = [
        [
            features.alpha_energy, features.alpha_energy_ratio,
            features.beta_energy, features.beta_energy_ratio,
            features.theta_energy, features.theta_energy_ratio,
            features.delta_energy, features.delta_energy_ratio
        ],
        [
            features.mean
        ]
    ]
    window = 100
    return __case(features_callbacks=features_callbacks, window=window)
    # 0.7535849056603774
    # FINISHED
    # 1 / 10
    # 0.7099427753934192
    # FINISHED
    # 2 / 10
    # 0.7003636363636364
    # FINISHED
    # 3 / 10
    # 0.6933579335793358
    # FINISHED
    # 4 / 10
    # 0.6708715596330275
    # FINISHED
    # 5 / 10
    # 0.7691751085383502
    # FINISHED
    # 6 / 10
    # 0.7138181818181818
    # FINISHED
    # 7 / 10
    # 0.6848899012908124
    # FINISHED
    # 8 / 10
    # 0.6535460992907801
    # FINISHED
    # 9 / 10
    # 0.6908339594290007
    # FINISHED
    # 10 / 10


@timer
def case_4():
    features_callbacks = [
        [
            features.root_mean_square, features.variance
        ],
        [

        ]
    ]
    window = 500
    return __case(features_callbacks=features_callbacks, window=window)
    # 0.759937106918239
    # FINISHED 1/10
    # 0.6765617548879351
    # FINISHED 2/10
    # 0.6678787878787878
    # FINISHED 3/10
    # 0.7177736777367774
    # FINISHED 4/10
    # 0.5681065239551478
    # FINISHED 5/10
    # 0.7906415822479498
    # FINISHED 6/10
    # 0.654969696969697
    # FINISHED 7/10
    # 0.7010883320678309
    # FINISHED 8/10
    # 0.6358156028368794
    # FINISHED 9/10
    # 0.6887678437265214
    # FINISHED 10/10
    # INFO: Method: case_4 finished with time 8088.924752950668 seconds
    # {'features_callbacks': [[<function root_mean_square at 0x7f984b2800d0>, <function variance at 0x7f983391bc80>], []], 'window': 500, 'model_config': {'input_amount': 256, 'output_amount': 3, 'hidden1': 32, 'hidden': 18, 'has_normalization': False, 'epochs': 5, 'batch_size': 256}, 'trained': 135, 'tested': 10, 'evaluates': [0.759937106918239, 0.6765617548879351, 0.6678787878787878, 0.7177736777367774, 0.5681065239551478, 0.7906415822479498, 0.654969696969697, 0.7010883320678309, 0.6358156028368794, 0.6887678437265214]}
    #
    # Process finished with exit code 0


@timer
def case_5():
    features_callbacks = [
        [
            features.energy, features.entropy, features.variation, features.root_mean_square
        ],
        [
        ]
    ]
    window = 300
    return __case(features_callbacks=features_callbacks, window=window)
    # 0.7535849056603774
    # FINISHED 1/10
    # 0.7099427753934192
    # FINISHED 2/10
    # 0.7003636363636364
    # FINISHED 3/10
    # 0.6933579335793358
    # FINISHED 4/10
    # 0.6708715596330275
    # FINISHED 5/10
    # 0.7691751085383502
    # FINISHED 6/10
    # 0.7138181818181818
    # FINISHED 7/10
    # 0.6848899012908124
    # FINISHED 8/10
    # 0.6535460992907801
    # FINISHED 9/10
    # 0.6908339594290007
    # FINISHED 10/10


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
def c8():
    registered_time = u_timer.get_time()
    create_data_directory(registered_time)
    create_training_directory(registered_time)
    create_test_directory(registered_time)
    f = [[features.mean, features.median], [features.max_peak]]
    window = 100
    print("TRAINING: ")
    l = 0
    for i in f:
        l += len(i)
    print(l)
    model = ann_m.Model(input_amount=l, output_amount=3, hidden1=48, hidden2=27, has_normalization=False)
    cls = ann_c.Classifier(model)
    counter = 0
    for data_set in data.examinations_data_set(
            training_signals, training_hypnogram, f, window, True, registered_time
    ):
        f_train, l_train = data.split(data_set)
        model.train(f_train, l_train, epochs=7, batch_size=256)
        counter += 1
        print("FINISHED {}/{}".format(counter, len(training_signals)))
    print("TEST: ")
    for data_set in data.examinations_data_set(
            test_signals, test_hypnogram, f, window, False, registered_time
    ):
        f_test, l_test = data.split(data_set)
        evaluate = cls.evaluate2(f_test, l_test)
        print(evaluate)
    model.save(registered_time)
    # TEST:
    # 0.7907169811320754
    # 0.749451597520267
    # 0.707260606060606
    # 0.7367527675276753
    # 0.6979357798165138
    # 0.8056560540279788
    # 0.7053333333333334
    # 0.6789800050620096
    # 0.6672695035460993
    # 0.7024417731029301
