from src.logger.messages import *
from src.utils.util import *
import src.logger.logger as logger


def __load_data_from_file(path):
    data = []
    try:
        logger.info(AASM_LOAD_START_INFO)
        with open(path, "r") as file:
            data = file.readlines()
            logger.info(AASM_LOAD_END_INFO)
    except FileNotFoundError:
        logger.error(AASM_LOAD_ERROR)

    return data


def __filter_data(data):
    # removed '\n' from hypno values e.g. "5\n" -> "5"
    filtered = []
    if data:
        filtered = list(map(lambda x: x.strip(), data))
    return filtered


def __split_data(data):
    title = ''
    hypnogram_data = []
    if data:
        title = data[0]
        hypnogram_data = data[1:]
    return title, hypnogram_data


def __convert_sleep_stages(hypnogram):
    a = []
    for s in hypnogram:
        if s is '5':
            a.append(AWAKE)
        elif s is '3' or s is '4':
            a.append(REM)
        elif s is '1' or s is '2':
            a.append(NON_REM)
        else:
            a.append(UNKNOWN)
    return a


def __extend_data(data):
    extended_data = []
    for sample in data:
        for i in range(1000):
            extended_data.append(sample)
    return extended_data


def load(path, convert=True, extend=True):
    hypnogram = __load_data_from_file(path)
    filtered = __filter_data(hypnogram)
    _, hypnogram_data = __split_data(filtered)
    if convert:
        hypnogram_data = __convert_sleep_stages(hypnogram_data)
    if extend:
        hypnogram_data = __extend_data(hypnogram_data)
    return hypnogram_data