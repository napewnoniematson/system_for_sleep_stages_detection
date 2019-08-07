from src.utils.util import *


class RK:

    def __init__(self, path):
        self.__path = path
        self.hypnogram = load(path)


def __load_data_from_file(path):
    data = []
    try:
        with open(path, "r") as file:
            data = file.readlines()
    except FileNotFoundError:
        print("hypnogram load err")
    return data


def __convert_stages(hypnogram):
    return [stage(i) for i in hypnogram]


def __filter_data(data):
    # removed '\n' from hypno values e.g. "5\n" -> "5"
    return list(map(lambda x: x.strip(), data)) if data else []


def __split_data(data):
    """ returning title and hypnogram_data"""
    return (data[0], data[1:]) if data else ('', [])


def unknown(i):
    if movement_time(i):
        return True
    return i is '-2' or i is '-3'


def movement_time(i):
    return i is '-1'


def awake(i):
    return i is '5'


def rem(i):
    return i is '3' or i is '4'


def non_rem(i):
    return i is '0' or i is '1' or i is '2'


def stage(i):
    return UNKNOWN if unknown(i) else AWAKE if awake(i) else REM if rem(i) else NON_REM if non_rem(i) else UNKNOWN


def __extend_data(data):
    extended_data = []
    for sample in data:
        for i in range(1000):
            extended_data.append(sample)
    return extended_data


def load(path):
    hypnogram = __load_data_from_file(path)
    filtered = __filter_data(hypnogram)
    _, hypnogram_data = __split_data(filtered)
    hypnogram_data = __convert_stages(hypnogram_data)
    hypnogram_data = __extend_data(hypnogram_data)
    return hypnogram_data
