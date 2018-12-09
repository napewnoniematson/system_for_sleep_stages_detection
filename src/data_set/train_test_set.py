from src.utils.util import *


def prepare_train_test_sets(data_set, training_part_percentage=TRAINING_PART_PERCENTAGE_DEFAULT):
    extreme_shuffle_data_set(data_set)
    split_index = int(len(data_set) * training_part_percentage / 100)
    training = data_set[:split_index]
    test = data_set[split_index:]
    l_train, f_train = _split_data_set_numpy(training)
    l_test, f_test = _split_data_set_numpy(test)
    return l_train, f_train, l_test, f_test


def _split_data_set_numpy(data_set):
    import numpy as np
    l_data, f_data = _split_data_set(data_set)
    return np.array(l_data), np.array(f_data)


def _split_data_set(data_set):
    l_data = []
    f_data = []
    for data in data_set:
        l_data.append(data[0])
        f_data.append(data[1])
    return l_data, f_data


def extreme_shuffle_data_set(data_set):
    import random
    for _ in range(3):
        random.shuffle(data_set)
