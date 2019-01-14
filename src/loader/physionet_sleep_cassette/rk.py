import pyedflib
import numpy as np
from src.utils.util import *
from src.logger.messages import *
import src.logger.logger as logger


class RK:

    def __init__(self, path):
        self.__path = path
        self.__edf = self.__load_edf_data(path)
        self.__indices = self.__edf.readAnnotations()[0]
        self.__stages = self.__edf.readAnnotations()[2]
        self.__stages = [stage(i) for i in self.__stages]
        self.hypnogram = np.zeros(int(self.__indices[-1] * 1000), np.int)
        for i in range(len(self.__stages) - 1):
            self.hypnogram[int(self.__indices[i]) * 1000:int(self.__indices[i + 1]) * 1000] = self.__stages[i]

    @staticmethod
    def __load_edf_data(path):
        logger.info(EEG_LOAD_START_INFO)
        edf = None
        try:
            edf = pyedflib.EdfReader(path)
            logger.info(EEG_LOAD_END_INFO)
        except OSError:
            logger.error(EEG_LOAD_ERROR)
        return edf


def unknown(i):
    if 'Movement time' in i:
        return True
    return i.split(' ')[2] is '?'


def awake(i):
    return i.split(' ')[2] is 'W'


def rem(i):
    return i.split(' ')[2] is 'R' or i.split(' ')[2] is '1'


def nrem(i):
    return i.split(' ')[2] is '2' or i.split(' ')[2] is '3' or i.split(' ')[2] is '4'


def stage(i):
    return -1 if unknown(i) else 0 if awake(i) else 1 if rem(i) else 2 if nrem(i) else -1
