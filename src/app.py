from src.logger.messages import *
import src.logger.logger as logger
import src.case as test_case


def demo_show():
    import src.figure as f
    import numpy as np

    a1 = [[28810, 966, 179], [188, 2195, 362], [1454, 1190, 4406]]
    a2 = [[25796, 2753, 1226], [286, 1464, 1235], [3619, 429, 5132]]
    a3 = [[27647, 1207, 36], [1044, 2437, 74], [2576, 2716, 3513]]
    a4 = [[27765, 383, 37], [1630, 2236, 184], [2541, 3638, 2236]]
    a5 = [[23299, 3003, 23], [1772, 3836, 167], [433, 5054, 1653]]
    a6 = [[31106, 588, 196], [416, 1775, 689], [1384, 970, 4336]]
    a7 = [[22560, 4911, 1974], [2767, 2584, 394], [2111, 3053, 896]]
    a8 = [[26482, 372, 206], [3485, 506, 224], [7297, 372, 566]]
    a9 = [[22878, 3750, 1017], [502, 5461, 1312], [627, 1686, 5067]]  # 79%
    a0 = [[26831, 403, 351], [8013, 250, 62], [3423, 384, 213]]  # 68%

    ctr = 0
    a = [a0, a1, a2, a3, a4, a5, a6, a7, a8, a9]
    for _a in a:
        # f.save_to_file_confusion_matrix(np.asarray(_a), ['AWAKE', 'REM', 'NREM'], '/home/pc/mgr/cm_ex/cm_ex_{}.png'.
        #                                 format(ctr), normalize=False).clf()
        # f.save_to_file_confusion_matrix(np.asarray(_a), ['AWAKE', 'REM', 'NREM'],
        #                                 '/home/pc/mgr/cm_ex/cm_ex_{}_norm.png'.format(ctr), normalize=True).clf()
        f.show_confusion_matrix(np.asarray(_a), ['AWAKE', 'REM', 'NREM'], normalize=False).clf()
        f.show_confusion_matrix(np.asarray(_a), ['AWAKE', 'REM', 'NREM'], normalize=True).clf()
        ctr += 1


def demo_save():
    import src.figure as f
    import numpy as np

    a1 = [[28810, 966, 179], [188, 2195, 362], [1454, 1190, 4406]]
    a2 = [[25796, 2753, 1226], [286, 1464, 1235], [3619, 429, 5132]]
    a3 = [[27647, 1207, 36], [1044, 2437, 74], [2576, 2716, 3513]]
    a4 = [[27765, 383, 37], [1630, 2236, 184], [2541, 3638, 2236]]
    a5 = [[23299, 3003, 23], [1772, 3836, 167], [433, 5054, 1653]]
    a6 = [[31106, 588, 196], [416, 1775, 689], [1384, 970, 4336]]
    a7 = [[22560, 4911, 1974], [2767, 2584, 394], [2111, 3053, 896]]
    a8 = [[26482, 372, 206], [3485, 506, 224], [7297, 372, 566]]
    a9 = [[22878, 3750, 1017], [502, 5461, 1312], [627, 1686, 5067]]  # 79%
    a0 = [[26831, 403, 351], [8013, 250, 62], [3423, 384, 213]]  # 68%

    ctr = 0
    a = [a0, a1, a2, a3, a4, a5, a6, a7, a8, a9]
    for _a in a:
        f.save_to_file_confusion_matrix(np.asarray(_a), ['AWAKE', 'REM', 'NREM'], '/home/pc/mgr/cm_ex/cm_ex_{}.png'.
                                        format(ctr), normalize=False).clf()
        f.save_to_file_confusion_matrix(np.asarray(_a), ['AWAKE', 'REM', 'NREM'],
                                        '/home/pc/mgr/cm_ex/cm_ex_{}_norm.png'.format(ctr), normalize=True).clf()
        ctr += 1


def sound_alert():
    import os
    duration = 0.2  # seconds
    freq = [200 * 1.4, 220 * 1.6, 240 * 1.2, 260 * 1.3, 280 * 1.4, 300 * 1.1]  # Hz
    for f in freq:
        os.system('play -nq -t alsa synth {} sine {}'.format(duration, f))


if __name__ == '__main__':
    logger.info(START_APP)
    import time
    s0 = time.time()
    for i in range(44):
        test_case.run(a=i)
    sound_alert()
    s1 = time.time()
    print("TIME: ", s1 - s0)








