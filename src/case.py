import time

import numpy as np

from src.decorator.timer import timer
import src.figure as fig
import src.utils.timer as u_timer
import src.loader.physionet_sleep_cassette.titles as titles
import src.feature_pack.feature as features
import src.feature_pack.pyegg_features as pyeeg_features
import src.ai.ann.model as ann_m
import src.ai.ann.classifier as ann_c
import src.data_set.data as data
import src.file_system.file_manager as file_m
from src.file_system.directory_manager import *
from src.model.ann_model_config import ANNModelConfig

create_main_directory()
create_model_directory()

default_model_config = ANNModelConfig()

physio_hypnogram, physio_signals = titles.get_physio_net(PHYSIONET_DIR)
de_mons_hypnogram, de_mons_signals = titles.get_de_mons(DE_MONS_DIR)


def _split_train_test(signals, hypnograms, x):
    """signals/hypnograms split to test per x rest is training"""
    training_hypnogram, training_signals, test_hypnogram, test_signals = [], [], [], []
    for i in range(len(signals)):
        if i % x == 0:
            test_hypnogram.append(hypnograms[i])
            test_signals.append(signals[i])
        else:
            training_hypnogram.append(hypnograms[i])
            training_signals.append(signals[i])
    return training_hypnogram, training_signals, test_hypnogram, test_signals


def _physio_split_data():
    return _split_train_test(physio_signals, physio_hypnogram, 8)


def _de_mons_split_data():
    return _split_train_test(de_mons_signals, de_mons_hypnogram, 6)


# SOURCE
PHYSIO = 0
DE_MONS = 1


def _choose_data(source_no):
    if PHYSIO == source_no:
        return _physio_split_data()
    if DE_MONS == source_no:
        return _de_mons_split_data()
    return [], [], [], []


def __case(source_no, features_callbacks, window, model_config=default_model_config):
    # start time
    registered_time = u_timer.get_time()
    # create directories
    create_data_directory(registered_time)
    create_training_directory(registered_time)
    create_test_directory(registered_time)
    create_predictions_directory(registered_time)
    # input data
    training_hypnogram, training_signals, test_hypnogram, test_signals = _choose_data(source_no)
    # amount of features
    input_amount = 0
    for callback in features_callbacks:
        input_amount += len(callback)
    model_config.input_amount = input_amount
    # prepare model
    print(model_config)
    model = ann_m.Model(model_config)
    # prepare classifier
    cls = ann_c.Classifier(model)
    # training part - calculate features + train model
    train_counter = 0
    start_training_time = time.time()
    for data_set in data.examinations_data_set(
            training_signals, training_hypnogram, features_callbacks, window, True, registered_time
    ):
        f_train, l_train = data.split(data_set)
        model.train(f_train, l_train)
        train_counter += 1
        print("FINISHED {}/{}".format(train_counter, len(training_signals)))
    # testing part - calculate features + evaluate model
    stop_training_time = time.time()
    print("EVALUATING")
    evaluates = []
    ctr = 0
    start_testing_time = time.time()
    for data_set in data.examinations_data_set(
            test_signals, test_hypnogram, features_callbacks, window, False, registered_time
    ):
        f_test, l_test = data.split(data_set)
        evaluate, predictions_arr = cls.evaluate2(f_test, l_test, model_config.output_amount)
        # print(evaluate)
        evaluates.append(evaluate)
        file_m.save(PREDICTIONS_FILE_DIR.format(registered_time, test_signals[ctr][0:-4]), [predictions_arr])
        ctr = ctr + 1
    stop_testing_time = time.time()
    # save model
    model.save(registered_time)
    # save confusion matrix files
    eval_ctr = 0
    for ev in evaluates:
        cm = np.asarray(ev['matrix'])
        fig.save_to_file_confusion_matrix(cm, ['AWAKE', 'REM', 'NREM'],
                                          PLOT_IMAGE_FILE_DIR.format(registered_time, eval_ctr), normalize=False).clf()
        fig.save_to_file_confusion_matrix(cm, ['AWAKE', 'REM', 'NREM'],
                                          NORMALIZED_PLOT_IMAGE_FILE_DIR.format(registered_time, eval_ctr),
                                          normalize=True).clf()
        eval_ctr = eval_ctr + 1
    result = {
        'source_no': source_no,
        'features_callbacks': features_callbacks,
        'window': window,
        'model_config': model_config,
        'trained': train_counter,
        'tested': len(evaluates),
        'training_time': stop_training_time - start_training_time,
        'testing_time': stop_testing_time - start_testing_time,
        'evaluates': evaluates
    }
    file_m.save(RESULT_FILE_DIR.format(registered_time), result.items())
    return result


@timer
def case_10_reworked():
    model_config = ANNModelConfig()
    features_callbacks = [
        [
            features.energy, features.entropy, features.variation, features.root_mean_square
        ],
        [
        ]
    ]
    window = 100
    return __case(source_no=DE_MONS, features_callbacks=features_callbacks, window=window, model_config=model_config)


@timer
def case_9_reworked():
    model_config = ANNModelConfig()
    features_callbacks = [
        [
            features.energy, features.entropy, features.variation, features.root_mean_square
        ],
        [
        ]
    ]
    window = 200
    return __case(source_no=DE_MONS, features_callbacks=features_callbacks, window=window, model_config=model_config)


@timer
def case_8_reworked():
    model_config = ANNModelConfig()
    features_callbacks = [
        [
            features.entropy, features.root_mean_square
        ],
        [
            features.variance
        ]
    ]
    window = 200
    return __case(source_no=DE_MONS, features_callbacks=features_callbacks, window=window, model_config=model_config)


@timer
def case_7_reworked():
    model_config = ANNModelConfig()
    features_callbacks = [
        [
            features.entropy, features.root_mean_square
        ],
        [
            features.variance, features.alpha_energy
        ]
    ]
    window = 200
    return __case(source_no=DE_MONS, features_callbacks=features_callbacks, window=window, model_config=model_config)


# 122/31
@timer
def case_5_reworked():
    model_config = ANNModelConfig()
    features_callbacks = [
        [
            features.energy, features.entropy, features.variation, features.root_mean_square
        ],
        [
        ]
    ]
    window = 300
    return __case(source_no=DE_MONS, features_callbacks=features_callbacks, window=window, model_config=model_config)


@timer
def case_4_reworked():
    model_config = ANNModelConfig()
    features_callbacks = [
        [
            features.root_mean_square, features.variance
        ],
        [

        ]
    ]
    window = 500
    return __case(source_no=DE_MONS, features_callbacks=features_callbacks, window=window, model_config=model_config)


def case_pyeeg_5():
    """
    seems to be good
    """
    model_config = ANNModelConfig()
    features_callbacks = [
        [
            pyeeg_features.petrosian_fractal_dimension
        ],
        [
        ]
    ]
    window = 200
    return __case(source_no=DE_MONS, features_callbacks=features_callbacks, window=window, model_config=model_config)


def case_pyeeg_6():
    model_config = ANNModelConfig()
    features_callbacks = [
        [
            pyeeg_features.fisher_info
        ],
        [
        ]
    ]
    window = 200
    return __case(source_no=DE_MONS, features_callbacks=features_callbacks, window=window, model_config=model_config)


@timer
def case_pyeeg_7():
    """time consuming, but good?"""
    model_config = ANNModelConfig()
    features_callbacks = [
        [
            pyeeg_features.permutation_entropy
        ],
        [
        ]
    ]
    window = 200
    return __case(source_no=DE_MONS, features_callbacks=features_callbacks, window=window, model_config=model_config)


def case_pyeeg_9():
    """fast"""
    model_config = ANNModelConfig()
    features_callbacks = [
        [
            pyeeg_features.svd_entropy
        ],
        [
        ]
    ]
    window = 200
    return __case(source_no=DE_MONS, features_callbacks=features_callbacks, window=window, model_config=model_config)


def case_pyeeg_12():
    """
    seems to be good (can not detect REM / check 2019.03.20 09:40)
    PHYSIO
    """
    model_config = ANNModelConfig()
    features_callbacks = [
        [
            pyeeg_features.petrosian_fractal_dimension
        ],
        [
        ]
    ]
    window = 200
    return __case(source_no=PHYSIO, features_callbacks=features_callbacks, window=window, model_config=model_config)


def case_pyeeg_13():
    """
    seems to be good
    PHYSIO
    """
    model_config = ANNModelConfig()
    features_callbacks = [
        [
            pyeeg_features.petrosian_fractal_dimension, features.root_mean_square, features.variance
        ],
        [
        ]
    ]
    window = 200
    return __case(source_no=PHYSIO, features_callbacks=features_callbacks, window=window, model_config=model_config)


def case_pyeeg_14():
    """
    seems to be good
    PHYSIO
    """
    model_config = ANNModelConfig()
    features_callbacks = [
        [
            pyeeg_features.petrosian_fractal_dimension, features.root_mean_square, features.variance
        ],
        [
        ]
    ]
    window = 200
    return __case(source_no=DE_MONS, features_callbacks=features_callbacks, window=window, model_config=model_config)


def case_pyeeg_15():
    model_config = ANNModelConfig()
    features_callbacks = [
        [
            pyeeg_features.power_delta, pyeeg_features.power_theta, pyeeg_features.power_alpha,
            pyeeg_features.power_beta, pyeeg_features.power_delta_ratio, pyeeg_features.power_theta_ratio,
            pyeeg_features.power_alpha_ratio, pyeeg_features.power_beta_ratio
        ],
        [
        ]
    ]
    window = 200
    return __case(source_no=DE_MONS, features_callbacks=features_callbacks, window=window, model_config=model_config)


def case_pyeeg_16():
    model_config = ANNModelConfig()
    features_callbacks = [
        [
            pyeeg_features.power_delta, pyeeg_features.power_theta, pyeeg_features.power_alpha,
            pyeeg_features.power_beta, pyeeg_features.power_delta_ratio, pyeeg_features.power_theta_ratio,
            pyeeg_features.power_alpha_ratio, pyeeg_features.power_beta_ratio
        ],
        [
        ]
    ]
    window = 200
    return __case(source_no=PHYSIO, features_callbacks=features_callbacks, window=window, model_config=model_config)


def case_pyeeg_17():
    model_config = ANNModelConfig()
    features_callbacks = [
        [
            # features.variance, features.root_mean_square, features.median, pyeeg_features.petrosian_fractal_dimension
            pyeeg_features.petrosian_fractal_dimension, features.root_mean_square, features.variance

        ],
        [
            # features.median#, features.root_mean_square, features.variance, pyeeg_features.petrosian_fractal_dimension
        ]
    ]
    window = 200
    return __case(source_no=PHYSIO, features_callbacks=features_callbacks, window=window, model_config=model_config)


def case_pyeeg_18():
    model_config = ANNModelConfig()
    features_callbacks = [
        [
            pyeeg_features.hjorth_mobility, pyeeg_features.hjorth_complexity

        ],
        [
        ]
    ]
    window = 100
    return __case(source_no=PHYSIO, features_callbacks=features_callbacks, window=window, model_config=model_config)
