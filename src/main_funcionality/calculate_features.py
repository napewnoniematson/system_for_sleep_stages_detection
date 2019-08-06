import src.utils.timer as u_timer
import src.loader.titles as titles
import src.feature_pack.feature as features
import src.feature_pack.pyegg_features as pyeeg_features
import src.data_set.data as data
import src.file_system.file_manager as file_m
from src.file_system.directory_manager import *

# SOURCE
PHYSIO = 0
DE_MONS = 1

# FILE NAMES
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


def calculate(source_no, features_callbacks, window):
    """ Calculate features and save them to file with csv format"""
    registered_time = u_timer.get_time()
    create_data_directory(registered_time)
    create_training_directory(registered_time)
    create_test_directory(registered_time)
    create_predictions_directory(registered_time)
    training_hypnogram, training_signals, test_hypnogram, test_signals = _choose_data(source_no)
    print(features_callbacks)
    print("TRAINING")
    ctr = 0
    for _ in data.examinations_data_set(
            training_signals, training_hypnogram, features_callbacks, window, True, registered_time
    ):
        ctr += 1
        print("%d/%d" % (ctr, len(training_signals)))

    print("TEST")
    ctr = 0
    for _ in data.examinations_data_set(
            test_signals, test_hypnogram, features_callbacks, window, False, registered_time
    ):
        ctr += 1
        print("%d/%d" % (ctr, len(test_signals)))
    file_m.save(RESULT_FILE_DIR.format(registered_time), features_callbacks)



def calculate_ratio():
    # registered_time = u_timer.get_time()
    # create_data_directory(registered_time)
    # create_training_directory(registered_time)
    # create_test_directory(registered_time)
    #
    # alpha_energy = None
    # alpha_fft_energy = None
    # beta_energy = None
    # beta_fft_energy = None
    # theta_energy = None
    # theta_fft_energy = None
    # delta_energy = None
    # delta_fft_energy = None
    pass


def calculate_500_physio(features_callbacks):
    window = 200
    return calculate(source_no=PHYSIO, features_callbacks=features_callbacks, window=window)


def calculate_500_physio_0():
    return calculate_500_physio([[], [features.root_mean_square]])


def calculate_500_physio_1():
    return calculate_500_physio([[], [features.max_peak]])


def calculate_500_physio_2():
    return calculate_500_physio([[], [features.min_peak]])


def calculate_500_physio_3():
    return calculate_500_physio([[], [features.mean]])


def calculate_500_physio_4():
    return calculate_500_physio([[], [features.harmonic_mean]])


def calculate_500_physio_5():
    return calculate_500_physio([[], [features.variance]])


def calculate_500_physio_6():
    return calculate_500_physio([[], [features.variation]])


def calculate_500_physio_7():
    return calculate_500_physio([[], [features.median]])


def calculate_500_physio_8():
    return calculate_500_physio([[], [features.energy]])


def calculate_500_physio_9():
    return calculate_500_physio([[], [features.alpha_energy]])


def calculate_500_physio_10():
    return calculate_500_physio([[], [features.beta_energy]])


def calculate_500_physio_11():
    return calculate_500_physio([[], [features.theta_energy]])


def calculate_500_physio_12():
    return calculate_500_physio([[], [features.delta_energy]])


def calculate_500_physio_13():
    return calculate_500_physio([[], [features.alpha_energy_ratio]])


def calculate_500_physio_14():
    return calculate_500_physio([[], [features.beta_energy_ratio]])


def calculate_500_physio_15():
    return calculate_500_physio([[], [features.theta_energy_ratio]])


def calculate_500_physio_16():
    return calculate_500_physio([[], [features.delta_energy_ratio]])


def calculate_500_physio_17():
    return calculate_500_physio([[], [features.sig_pow]])


def calculate_500_physio_18():
    return calculate_500_physio([[], [features.sig_pow_sk]])


def calculate_500_physio_19():
    return calculate_500_physio([[], [features.alpha_fft_energy]])


def calculate_500_physio_20():
    return calculate_500_physio([[], [features.beta_fft_energy]])


def calculate_500_physio_21():
    return calculate_500_physio([[], [features.theta_fft_energy]])


def calculate_500_physio_22():
    return calculate_500_physio([[], [features.delta_fft_energy]])


def calculate_500_physio_23():
    return calculate_500_physio([[], [features.alpha_fft_energy_ratio]])


def calculate_500_physio_24():
    return calculate_500_physio([[], [features.beta_fft_energy_ratio]])


def calculate_500_physio_25():
    return calculate_500_physio([[], [features.theta_fft_energy_ratio]])


def calculate_500_physio_26():
    return calculate_500_physio([[], [features.delta_fft_energy_ratio]])


def calculate_500_physio_27():
    return calculate_500_physio([[], [features.spectrogram2_energy]])


def calculate_500_physio_28():
    return calculate_500_physio([[], [features.spectrogram2_sig_pow]])


def calculate_500_physio_29():
    return calculate_500_physio([[], [features.spectrogram2_sig_pow_sk]])


def calculate_500_physio_30():
    return calculate_500_physio([[], [pyeeg_features.power_delta]])


def calculate_500_physio_31():
    return calculate_500_physio([[], [pyeeg_features.power_theta]])


def calculate_500_physio_32():
    return calculate_500_physio([[], [pyeeg_features.power_beta]])


def calculate_500_physio_33():
    return calculate_500_physio([[], [pyeeg_features.power_alpha]])


def calculate_500_physio_34():
    return calculate_500_physio([[], [pyeeg_features.power_delta_ratio]])


def calculate_500_physio_35():
    return calculate_500_physio([[], [pyeeg_features.power_theta_ratio]])


def calculate_500_physio_36():
    return calculate_500_physio([[], [pyeeg_features.power_beta_ratio]])


def calculate_500_physio_37():
    return calculate_500_physio([[], [pyeeg_features.power_alpha_ratio]])


def calculate_500_physio_38():
    return calculate_500_physio([[], [pyeeg_features.hjorth_mobility]])


def calculate_500_physio_39():
    return calculate_500_physio([[], [pyeeg_features.hjorth_complexity]])


def calculate_500_physio_40():
    return calculate_500_physio([[], [pyeeg_features.petrosian_fractal_dimension]])


def calculate_500_physio_41():
    return calculate_500_physio([[], [pyeeg_features.fisher_info]])


def calculate_500_physio_42():
    return calculate_500_physio([[], [pyeeg_features.permutation_entropy]])


def calculate_500_physio_43():
    return calculate_500_physio([[], [pyeeg_features.svd_entropy]])


