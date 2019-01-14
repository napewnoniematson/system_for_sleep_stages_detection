# from src.data_set.full_set import *
# from src.data_set.train_test_set import *
from src.feature_pack.feature import *


def start():
    a = [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2]
    print(alpha_energy(a))
    print(alpha_energy_ratio(a))
