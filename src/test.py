from src.data_set.full_set import *
from src.data_set.train_test_set import *
import matplotlib.pyplot as plt
from src.loader.aasm import load, UNIVERSITE_DE_MONS_HYPNOGRAM_AASM

def start():
    hypnogram = load(UNIVERSITE_DE_MONS_HYPNOGRAM_AASM.format('subject4'))
    plt.plot(hypnogram)
    plt.show()



def model_t():
    examinations = [[x for x in range(40)] for _ in range(50000)]
    callbacks = [min, max]
    classes = [[s for s in "0112200200345120001234512023451200000123"] for _ in range(50000)]
    xd = [(e, c) for e, c in zip(examinations, classes)]
    ls = examinations_data_set(
        xd, callbacks, 1, StageMode.REM_VS_NREM.value
    )
    checked = [0 if i % 5 == 0 else 1 for i in range(50000)]
    print(checked)

    l1, f1, l2, f2 = train_test_data(ls, checked)
    print(f1)
    print(l1)
    print(f2)
    print(l2)
    from src.ai.ann.model import Model
    model = Model(f1, l1, has_normalization=False, epochs=5)


