import pyedflib
import numpy as np
from src.utils.util import *


class RK:

    def __init__(self, path):
        self.__path = path
        self.__edf = pyedflib.EdfReader(path)
        self.__indices = self.__edf.readAnnotations()[0]
        self.__stages = self.__edf.readAnnotations()[2]
        self.__convert_stages()
        self.__read_hypnogram()

    def __convert_stages(self):
        self.__stages = [stage(i) for i in self.__stages]

    def __read_hypnogram(self):
        times = 100  # expert registered stage per 1 sec, 1 sec = 100 samples
        self.hypnogram = np.zeros(int(self.__indices[-1] * times), np.int)
        for i in range(len(self.__stages) - 1):
            self.hypnogram[int(self.__indices[i]) * times:int(self.__indices[i + 1]) * times] = self.__stages[i]


def unknown(i):
    if movement_time(i):
        return True
    return i.split(' ')[2] is '?'


def movement_time(i):
    return 'Movement time' in i


def awake(i):
    return i.split(' ')[2] is 'W'


def rem(i):
    return i.split(' ')[2] is 'R' or i.split(' ')[2] is '1'


def non_rem(i):
    return i.split(' ')[2] is '2' or i.split(' ')[2] is '3' or i.split(' ')[2] is '4'


def stage(i):
    return UNKNOWN if unknown(i) else AWAKE if awake(i) else REM if rem(i) else NON_REM if non_rem(i) else UNKNOWN
