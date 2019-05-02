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
import src.diagnostic_reliability as dr
from src.file_system.directory_manager import *
from src.model.ann_model_config import ANNModelConfig


default_model_config = ANNModelConfig()
physio_hypnogram, physio_signals = titles.get_physio_net(PHYSIONET_DIR)
de_mons_hypnogram, de_mons_signals = titles.get_de_mons(DE_MONS_DIR)


def _split_train_test(signals, hypnograms, x):
    """
    Split signal and hypnogram to training and testing sets

    :param signals: list of signals (names)
    :param hypnograms: list of hypnogram (names)
    :param x: signal / hypnogram which is multiple of x is treated as train signal / hypnogram,
     rest are treated as training signal / hypnogram
    :return: training_hypnogram, training_signals, test_hypnogram, test_signals
    """
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
    """
    Split signal and hypnogram to training and testing sets for physionet data

    :return: training_hypnogram, training_signals, test_hypnogram, test_signals
    """
    return _split_train_test(physio_signals, physio_hypnogram, 8)


def _de_mons_split_data():
    """
    Split signal and hypnogram to training and testing sets for 'de mons' data

    :return: training_hypnogram, training_signals, test_hypnogram, test_signals
    """
    return _split_train_test(de_mons_signals, de_mons_hypnogram, 6)


# SOURCE
PHYSIO = 0
DE_MONS = 1


def _choose_data(source_no):
    """
    Function allow choosing input data by passing number parameter

    :param source_no: number which represent source with data, available PHYSIO = 0 DE_MONS = 1
    :return: training_hypnogram, training_signals, test_hypnogram, test_signals
    """
    if PHYSIO == source_no:
        return _physio_split_data()
    if DE_MONS == source_no:
        return _de_mons_split_data()
    return [], [], [], []


def __case(source_no, features_callbacks, window, model_config=default_model_config):
    """
    Research program main flow, which allows fast preparing test cases

    :param source_no: number which represent source with data, available PHYSIO = 0 DE_MONS = 1
    :param features_callbacks: list of callbacks to methods which calculates features
    :param window: signals are split to many segments which contains same number of samples, this number is equal window
    :param model_config: data with information about used structure of Artificial Neural Network
    :return: result, its a dictionary with data collected during method call, dictionary contains
        source_no, features_callbacks, window,model_config, trained, tested, training_time, testing_time, evaluates
    """
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
    evaluates, diagnostics = [], []
    ctr = 0
    start_testing_time = time.time()
    for data_set in data.examinations_data_set(
            test_signals, test_hypnogram, features_callbacks, window, False, registered_time
    ):
        f_test, l_test = data.split(data_set)
        evaluate, diagnostic, predictions_arr = cls.evaluate2(f_test, l_test, model_config.output_amount)
        # print(evaluate)
        evaluates.append(evaluate)
        diagnostics.append(diagnostic)
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
                                          PLOT_IMAGE_FILE_DIR.format(registered_time, eval_ctr, test_signals[eval_ctr]),
                                          normalize=False).clf()
        fig.save_to_file_confusion_matrix(cm, ['AWAKE', 'REM', 'NREM'],
                                          NORMALIZED_PLOT_IMAGE_FILE_DIR.format(registered_time, eval_ctr,
                                                                                test_signals[eval_ctr]),
                                          normalize=True).clf()
        eval_ctr = eval_ctr + 1

    total_tp, total_tn, total_fn, total_fp = 0, 0, 0, 0
    for diag in diagnostics:
        total_tp = total_tp + diag['true_positive']
        total_tn = total_tn + diag['true_negative']
        total_fn = total_fn + diag['false_negative']
        total_fp = total_fp + diag['false_positive']

    total_accuracy = dr.accuracy(total_tp, total_fp, total_fn, total_tn)
    total_sensitivity = dr.sensitivity(total_tp, total_fn)
    total_specificity = dr.specificity(total_fp, total_tn)
    total_ppv = dr.positive_predictive_value(total_tp, total_fp)
    total_npv = dr.negative_predictive_value(total_fn, total_tn)
    result = {
        'source_no': source_no,
        'features_callbacks': features_callbacks,
        'window': window,
        'model_config': model_config,
        'trained': train_counter,
        'tested': len(evaluates),
        'training_time': stop_training_time - start_training_time,
        'testing_time': stop_testing_time - start_testing_time,
        'total_diagnostic': {
            'total_accuracy': total_accuracy,
            'total_sensitivity': total_sensitivity,
            'total_specificity': total_specificity,
            'total_ppv': total_ppv,
            'total_npv': total_npv
        },
        'diagnostic': diagnostics,
        'evaluates': evaluates
    }
    # save results to file
    file_m.save(RESULT_FILE_DIR.format(registered_time), result.items())
    return result



@timer
def basic_time_test_de_mons(callback):
    model_config = ANNModelConfig()
    features_callbacks = [
        [
            callback

        ],
        [
        ]
    ]
    window = 200
    return __case(source_no=DE_MONS, features_callbacks=features_callbacks, window=window, model_config=model_config)


def basic_time_test_de_mons_0():
    """EEG: root_mean_squere"""
    return basic_time_test_de_mons(features.root_mean_square)


def basic_time_test_de_mons_1():
    """EEG: max_peak"""
    return basic_time_test_de_mons(features.max_peak)


def basic_time_test_de_mons_2():
    """EEG: min_peak"""
    return basic_time_test_de_mons(features.min_peak)


def basic_time_test_de_mons_3():
    """EEG: mean"""
    return basic_time_test_de_mons(features.mean)


def basic_time_test_de_mons_4():
    """EEG: harmonic_mean"""
    return basic_time_test_de_mons(features.harmonic_mean)


def basic_time_test_de_mons_5():
    """EEG: variance"""
    return basic_time_test_de_mons(features.variance)


def basic_time_test_de_mons_6():
    """EEG: variation"""
    return basic_time_test_de_mons(features.variation)


def basic_time_test_de_mons_7():
    """EEG: spectrogram"""
    return basic_time_test_de_mons(features.spectrogram)


def basic_time_test_de_mons_8():
    """EEG: median"""
    return basic_time_test_de_mons(features.median)


def basic_time_test_de_mons_9():
    """EEG: energy"""
    return basic_time_test_de_mons(features.energy)


def basic_time_test_de_mons_10():
    """EEG: alpha_energy"""
    return basic_time_test_de_mons(features.alpha_energy)


def basic_time_test_de_mons_11():
    """EEG: beta_energy"""
    return basic_time_test_de_mons(features.beta_energy)


def basic_time_test_de_mons_12():
    """EEG: theta_energy"""
    return basic_time_test_de_mons(features.theta_energy)


def basic_time_test_de_mons_13():
    """EEG: delta_energy"""
    return basic_time_test_de_mons(features.delta_energy)


def basic_time_test_de_mons_14():
    """EEG: alpha_energy_ratio"""
    return basic_time_test_de_mons(features.alpha_energy_ratio)


def basic_time_test_de_mons_15():
    """EEG: beta_energy_ratio"""
    return basic_time_test_de_mons(features.beta_energy_ratio)


def basic_time_test_de_mons_16():
    """EEG: theta_energy_ratio"""
    return basic_time_test_de_mons(features.theta_energy_ratio)


def basic_time_test_de_mons_17():
    """EEG: delta_energy_ratio"""
    return basic_time_test_de_mons(features.delta_energy_ratio)


def basic_time_test_de_mons_18():
    """EEG: sig_pow"""
    return basic_time_test_de_mons(features.sig_pow)


def basic_time_test_de_mons_19():
    """EEG: sig_pow_sk"""
    return basic_time_test_de_mons(features.sig_pow_sk)


def basic_time_test_de_mons_20():
    """EEG: alpha_fft_energy"""
    return basic_time_test_de_mons(features.alpha_fft_energy)


def basic_time_test_de_mons_21():
    """EEG: beta_fft_energy"""
    return basic_time_test_de_mons(features.beta_fft_energy)


def basic_time_test_de_mons_22():
    """EEG: theta_fft_energy"""
    return basic_time_test_de_mons(features.theta_fft_energy)


def basic_time_test_de_mons_23():
    """EEG: alpha_fft_energy_ratio"""
    return basic_time_test_de_mons(features.alpha_fft_energy_ratio)


def basic_time_test_de_mons_24():
    """EEG: beta_fft_energy_ratio"""
    return basic_time_test_de_mons(features.beta_fft_energy_ratio)


def basic_time_test_de_mons_25():
    """EEG: theta_fft_energy_ratio"""
    return basic_time_test_de_mons(features.theta_fft_energy_ratio)


def basic_time_test_de_mons_26():
    """EEG: delta_fft_energy_ratio"""
    return basic_time_test_de_mons(features.delta_fft_energy_ratio)


def basic_time_test_de_mons_27():
    """EEG: spectrogram2_energy"""
    return basic_time_test_de_mons(features.spectrogram2_energy)


def basic_time_test_de_mons_28():
    """EEG: spectrogram2_sig_pow"""
    return basic_time_test_de_mons(features.spectrogram2_sig_pow)


def basic_time_test_de_mons_29():
    """EEG: spectrogram2_sig_pow_sk"""
    return basic_time_test_de_mons(features.spectrogram2_sig_pow_sk)


# PYEEG
def basic_time_test_de_mons_30():
    """EEG: power_delta"""
    return basic_time_test_de_mons(pyeeg_features.power_delta)


def basic_time_test_de_mons_31():
    """EEG: power_theta"""
    return basic_time_test_de_mons(pyeeg_features.power_theta)


def basic_time_test_de_mons_32():
    """EEG: power_beta"""
    return basic_time_test_de_mons(pyeeg_features.power_beta)


def basic_time_test_de_mons_33():
    """EEG: power_alpha"""
    return basic_time_test_de_mons(pyeeg_features.power_alpha)


def basic_time_test_de_mons_34():
    """EEG: power_delta_ratio"""
    return basic_time_test_de_mons(pyeeg_features.power_delta_ratio)


def basic_time_test_de_mons_35():
    """EEG: power_theta_ratio"""
    return basic_time_test_de_mons(pyeeg_features.power_theta_ratio)


def basic_time_test_de_mons_36():
    """EEG: power_beta_ratio"""
    return basic_time_test_de_mons(pyeeg_features.power_beta_ratio)


def basic_time_test_de_mons_37():
    """EEG: power_alpha_ratio"""
    return basic_time_test_de_mons(pyeeg_features.power_alpha_ratio)


def basic_time_test_de_mons_38():
    """EEG: hjorth_mobility"""
    return basic_time_test_de_mons(pyeeg_features.hjorth_mobility)


def basic_time_test_de_mons_39():
    """EEG: hjorth_complexity"""
    return basic_time_test_de_mons(pyeeg_features.hjorth_complexity)


def basic_time_test_de_mons_40():
    """EEG: petrosian_fractal_dimension"""
    return basic_time_test_de_mons(pyeeg_features.petrosian_fractal_dimension)


def basic_time_test_de_mons_41():
    """EEG: fisher_info"""
    return basic_time_test_de_mons(pyeeg_features.fisher_info)


def basic_time_test_de_mons_42():
    """EEG: permutation_entropy"""
    return basic_time_test_de_mons(pyeeg_features.permutation_entropy)


def basic_time_test_de_mons_43():
    """EEG: svd_entropy"""
    return basic_time_test_de_mons(pyeeg_features.svd_entropy)


@timer
def basic_time_test_physio(callback):
    model_config = ANNModelConfig()
    features_callbacks = [
        [
            callback

        ],
        [
        ]
    ]
    window = 200
    return __case(source_no=PHYSIO, features_callbacks=features_callbacks, window=window, model_config=model_config)


def basic_time_test_physio_0():
    """EEG: root_mean_square"""
    return basic_time_test_physio(features.root_mean_square)


def basic_time_test_physio_1():
    """EEG: max_peak"""
    return basic_time_test_physio(features.max_peak)


def basic_time_test_physio_2():
    """EEG: min_peak"""
    return basic_time_test_physio(features.min_peak)


def basic_time_test_physio_3():
    """EEG: mean"""
    return basic_time_test_physio(features.mean)


def basic_time_test_physio_4():
    """EEG: harmonic_mean"""
    return basic_time_test_physio(features.harmonic_mean)


def basic_time_test_physio_5():
    """EEG: variance"""
    return basic_time_test_physio(features.variance)


def basic_time_test_physio_6():
    """EEG: variation"""
    return basic_time_test_physio(features.variation)


def basic_time_test_physio_7():
    """EEG: spectrogram"""
    return basic_time_test_physio(features.spectrogram)


def basic_time_test_physio_8():
    """EEG: median"""
    return basic_time_test_physio(features.median)


def basic_time_test_physio_9():
    """EEG: energy"""
    return basic_time_test_physio(features.energy)


def basic_time_test_physio_10():
    """EEG: alpha_energy"""
    return basic_time_test_physio(features.alpha_energy)


def basic_time_test_physio_11():
    """EEG: beta_energy"""
    return basic_time_test_physio(features.beta_energy)


def basic_time_test_physio_12():
    """EEG: theta_energy"""
    return basic_time_test_physio(features.theta_energy)


def basic_time_test_physio_13():
    """EEG: delta_energy"""
    return basic_time_test_physio(features.delta_energy)


def basic_time_test_physio_14():
    """EEG: alpha_energy_ratio"""
    return basic_time_test_physio(features.alpha_energy_ratio)


def basic_time_test_physio_15():
    """EEG: beta_energy_ratio"""
    return basic_time_test_physio(features.beta_energy_ratio)


def basic_time_test_physio_16():
    """EEG: theta_energy_ratio"""
    return basic_time_test_physio(features.theta_energy_ratio)


def basic_time_test_physio_17():
    """EEG: delta_energy_ratio"""
    return basic_time_test_physio(features.delta_energy_ratio)


def basic_time_test_physio_18():
    """EEG: sig_pow"""
    return basic_time_test_physio(features.sig_pow)


def basic_time_test_physio_19():
    """EEG: sig_pow_sk"""
    return basic_time_test_physio(features.sig_pow_sk)


def basic_time_test_physio_20():
    """EEG: alpha_fft_energy"""
    return basic_time_test_physio(features.alpha_fft_energy)


def basic_time_test_physio_21():
    """EEG: beta_fft_energy"""
    return basic_time_test_physio(features.beta_fft_energy)


def basic_time_test_physio_22():
    """EEG: theta_fft_energy"""
    return basic_time_test_physio(features.theta_fft_energy)


def basic_time_test_physio_23():
    """EEG: alpha_fft_energy_ratio"""
    return basic_time_test_physio(features.alpha_fft_energy_ratio)


def basic_time_test_physio_24():
    """EEG: beta_fft_energy_ratio"""
    return basic_time_test_physio(features.beta_fft_energy_ratio)


def basic_time_test_physio_25():
    """EEG: theta_fft_energy_ratio"""
    return basic_time_test_physio(features.theta_fft_energy_ratio)


def basic_time_test_physio_26():
    """EEG: delta_fft_energy_ratio"""
    return basic_time_test_physio(features.delta_fft_energy_ratio)


def basic_time_test_physio_27():
    """EEG: spectrogram2_energy"""
    return basic_time_test_physio(features.spectrogram2_energy)


def basic_time_test_physio_28():
    """EEG: spectrogram2_sig_pow"""
    return basic_time_test_physio(features.spectrogram2_sig_pow)


def basic_time_test_physio_29():
    """EEG: spectrogram2_sig_pow_sk"""
    return basic_time_test_physio(features.spectrogram2_sig_pow_sk)


# PYEGG
def basic_time_test_physio_30():
    """EEG: power_delta"""
    return basic_time_test_physio(pyeeg_features.power_delta)


def basic_time_test_physio_31():
    """EEG: power_theta"""
    return basic_time_test_physio(pyeeg_features.power_theta)


def basic_time_test_physio_32():
    """EEG: power_beta"""
    return basic_time_test_physio(pyeeg_features.power_beta)


def basic_time_test_physio_33():
    """EEG: power_alpha"""
    return basic_time_test_physio(pyeeg_features.power_alpha)


def basic_time_test_physio_34():
    """EEG: power_delta_ratio"""
    return basic_time_test_physio(pyeeg_features.power_delta_ratio)


def basic_time_test_physio_35():
    """EEG: power_theta_ratio"""
    return basic_time_test_physio(pyeeg_features.power_theta_ratio)


def basic_time_test_physio_36():
    """EEG: power_beta_ratio"""
    return basic_time_test_physio(pyeeg_features.power_beta_ratio)


def basic_time_test_physio_37():
    """EEG: power_alpha_ratio"""
    return basic_time_test_physio(pyeeg_features.power_alpha_ratio)


def basic_time_test_physio_38():
    """EEG: hjorth_mobility"""
    return basic_time_test_physio(pyeeg_features.hjorth_mobility)


def basic_time_test_physio_39():
    """EEG: hjorth_complexity"""
    return basic_time_test_physio(pyeeg_features.hjorth_complexity)


def basic_time_test_physio_40():
    """EEG: petrosian_fractal_dimension"""
    return basic_time_test_physio(pyeeg_features.petrosian_fractal_dimension)


def basic_time_test_physio_41():
    """EEG: fisher_info"""
    return basic_time_test_physio(pyeeg_features.fisher_info)


def basic_time_test_physio_42():
    """EEG: permutation_entropy"""
    return basic_time_test_physio(pyeeg_features.permutation_entropy)


def basic_time_test_physio_43():
    """EEG: svd_entropy"""
    return basic_time_test_physio(pyeeg_features.svd_entropy)


# MY FEATURES
    # basic_time_test_de_mons_0: 143.72357511520386
    # basic_time_test_de_mons_1: 126.6245048046112
    # basic_time_test_de_mons_2: 123.36828899383545
    # basic_time_test_de_mons_3: 343.2106053829193
    # basic_time_test_de_mons_4: 153.28992319107056
    # basic_time_test_de_mons_5: 861.0224967002869
    # basic_time_test_de_mons_6: 281.8074481487274
    # basic_time_test_de_mons_7: err
    # basic_time_test_de_mons_8: 140.12274861335754
    # basic_time_test_de_mons_9: 167.43756341934204
    # basic_time_test_de_mons_10: 386.79503893852234
    # basic_time_test_de_mons_11: 381.5411922931671
    # basic_time_test_de_mons_12: 380.0794503688812
    # basic_time_test_de_mons_13: 391.2507140636444
    # basic_time_test_de_mons_14: 1099.9526300430298
    # basic_time_test_de_mons_15: 1158.1956534385681
    # basic_time_test_de_mons_16: 1119.2417442798615
    # basic_time_test_de_mons_17: 1143.2096054553986
    # basic_time_test_de_mons_18: 167.6562819480896
    # basic_time_test_de_mons_19: 170.19502544403076
    # basic_time_test_de_mons_20: 393.2258071899414
    # basic_time_test_de_mons_21: 391.01372170448303
    # basic_time_test_de_mons_22: 397.8124327659607
    # basic_time_test_de_mons_23: 1245.8141016960144
    # basic_time_test_de_mons_24: 1190.1889419555664
    # basic_time_test_de_mons_25: 1172.0741460323334
    # basic_time_test_de_mons_26: 1157.4680964946747
    # basic_time_test_de_mons_27: 289.3271577358246
    # basic_time_test_de_mons_28: 289.62707710266113
    # basic_time_test_de_mons_29: 293.355934381485

    # PYEEG FEATURES
    # basic_time_test_de_mons_30: 154.0344307422638
    # basic_time_test_de_mons_31: 153.95497131347656
    # basic_time_test_de_mons_32: 154.0412483215332
    # basic_time_test_de_mons_33: 154.44360518455505
    # basic_time_test_de_mons_34: 154.41375136375427
    # basic_time_test_de_mons_35: 153.45395636558533
    # basic_time_test_de_mons_36: 157.2384970188141
    # basic_time_test_de_mons_37: 155.02633237838745
    # basic_time_test_de_mons_38: 229.47436332702637
    # basic_time_test_de_mons_39: 239.2857015132904
    # basic_time_test_de_mons_40: 176.86260628700256
    # basic_time_test_de_mons_41: 161.64627027511597
    # basic_time_test_de_mons_42: 2068.6068625450134
    # basic_time_test_de_mons_43: 161.38730144500732

    # PHYSIO
    # basic_time_test_physio_0: 1189.2667751312256
    # basic_time_test_physio_1: 1019.8352560997009
    # basic_time_test_physio_2: 1031.8725838661194
    # basic_time_test_physio_3: 3447.3305072784424
    # basic_time_test_physio_4: 1297.86128115654
    # basic_time_test_physio_5: 9225.565055131912
    # basic_time_test_physio_6: 1377.0041360855103
    # basic_time_test_physio_7: err
    # basic_time_test_physio_8: 1245.699406862259
    # basic_time_test_physio_9: 1476.5621535778046
    # basic_time_test_physio_10: 3683.527688264847
    # basic_time_test_physio_11: 3820.8386425971985
    # basic_time_test_physio_12: 4011.121281147003
    # basic_time_test_physio_13: 3810.3192677497864
    # basic_time_test_physio_14: too long
    # basic_time_test_physio_15: too long
    # basic_time_test_physio_16: too long
    # basic_time_test_physio_17: too long
    # basic_time_test_physio_18: 1718.0101916790009
    # basic_time_test_physio_19: 1536.8954167366028
    # basic_time_test_physio_20: 4183.404004812241
    # basic_time_test_physio_21: 3942.4722266197205
    # basic_time_test_physio_22: 3644.452351331711
    # basic_time_test_physio_23: too long
    # basic_time_test_physio_24: too long
    # basic_time_test_physio_25: too long
    # basic_time_test_physio_26: too long
    # basic_time_test_physio_27: 2639.911137342453
    # basic_time_test_physio_28: 2664.324657678604
    # basic_time_test_physio_29: 2703.1563200950623
    # basic_time_test_physio_30: 1335.1774685382843
    # basic_time_test_physio_31: 1327.127035856247
    # basic_time_test_physio_32: 1348.7438204288483
    # basic_time_test_physio_33: 1346.9565255641937
    # basic_time_test_physio_34: 1370.7704150676727
    # basic_time_test_physio_35: 1392.9937829971313
    # basic_time_test_physio_36: 1475.9210562705994
    # basic_time_test_physio_37: 1469.4200489521027
    # basic_time_test_physio_38: 2083.2481853961945
    # basic_time_test_physio_39: 2130.488794565201
    # basic_time_test_physio_40: 1342.9916791915894
    # basic_time_test_physio_41: 1500.8176662921906
    # basic_time_test_physio_42:: too long
    # basic_time_test_physio_43: 1529.4244935512543
