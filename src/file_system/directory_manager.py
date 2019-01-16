import os
from src.utils.util import *


def _create_directory(path):
    if not os.path.exists(path):
        try:
            os.mkdir(path)
        except OSError:
            raise
    return path


def create_main_directory():
    _create_directory(MAIN_FEATURE_DIR)


def create_model_directory():
    _create_directory(MAIN_MODEL_DIR)


def create_data_directory(date):
    _create_directory(DATE_FEATURE_DIR.format(date))


def create_training_directory(date):
    _create_directory(TRAINING_FEATURE_DIR.format(date))


def create_test_directory(date):
    _create_directory(TEST_FEATURE_DIR.format(date))
