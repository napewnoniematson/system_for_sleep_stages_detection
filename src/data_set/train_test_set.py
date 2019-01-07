from src.utils.util import *
from deprecated import deprecated


def train_test_data(data_set, amount=15):
    # training = [data_set[i] for i in range(amount)]
    # test = [data_set[i] for i in checked_examinations if i == 0]
    training = [data_set[i] for i in range(15)]
    test = [data_set[19-i] for i in range(5)]
    training = flatten_2_level_data_set(training)
    test = flatten_2_level_data_set(test)

    print("training: {}\ntest: {}".format(training, test))

    l_train, f_train = split_data_set_numpy(training)
    l_test, f_test = split_data_set_numpy(test)

    return l_train, f_train, l_test, f_test


def split_data_set_numpy(data_set):
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


def flatten_2_level_data_set(data_set):
    temp = []
    for examination in data_set:
        for pair in examination:
            temp.append(pair)
    return temp
