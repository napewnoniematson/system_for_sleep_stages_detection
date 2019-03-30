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


if __name__ == '__main__':
    logger.info(START_APP)
    # demo_show()
    print(test_case.basic_time_test_physio_43())

    # MY FEATURES
    # basic_time_test_de_mons_0: 143.72357511520386
    # basic_time_test_de_mons_1: 126.6245048046112
    # basic_time_test_de_mons_2: 123.36828899383545
    # basic_time_test_de_mons_3: 343.2106053829193
    # basic_time_test_de_mons_4: 153.28992319107056
    # basic_time_test_de_mons_5: 861.0224967002869
    # basic_time_test_de_mons_6: 281.8074481487274
    # basic_time_test_de_mons_7: err
    # basic_time_test_de_mons_8: 140.12274861335754
    # basic_time_test_de_mons_9: 167.43756341934204
    # basic_time_test_de_mons_10: 386.79503893852234
    # basic_time_test_de_mons_11: 381.5411922931671
    # basic_time_test_de_mons_12: 380.0794503688812
    # basic_time_test_de_mons_13: 391.2507140636444
    # basic_time_test_de_mons_14: 1099.9526300430298
    # basic_time_test_de_mons_15: 1158.1956534385681
    # basic_time_test_de_mons_16: 1119.2417442798615
    # basic_time_test_de_mons_17: 1143.2096054553986
    # basic_time_test_de_mons_18: 167.6562819480896
    # basic_time_test_de_mons_19: 170.19502544403076
    # basic_time_test_de_mons_20: 393.2258071899414
    # basic_time_test_de_mons_21: 391.01372170448303
    # basic_time_test_de_mons_22: 397.8124327659607
    # basic_time_test_de_mons_23: 1245.8141016960144
    # basic_time_test_de_mons_24: 1190.1889419555664
    # basic_time_test_de_mons_25: 1172.0741460323334
    # basic_time_test_de_mons_26: 1157.4680964946747
    # basic_time_test_de_mons_27: 289.3271577358246
    # basic_time_test_de_mons_28: 289.62707710266113
    # basic_time_test_de_mons_29: 293.355934381485

    # PYEEG FEATURES
    # basic_time_test_de_mons_30: 154.0344307422638
    # basic_time_test_de_mons_31: 153.95497131347656
    # basic_time_test_de_mons_32: 154.0412483215332
    # basic_time_test_de_mons_33: 154.44360518455505
    # basic_time_test_de_mons_34: 154.41375136375427
    # basic_time_test_de_mons_35: 153.45395636558533
    # basic_time_test_de_mons_36: 157.2384970188141
    # basic_time_test_de_mons_37: 155.02633237838745
    # basic_time_test_de_mons_38: 229.47436332702637
    # basic_time_test_de_mons_39: 239.2857015132904
    # basic_time_test_de_mons_40: 176.86260628700256
    # basic_time_test_de_mons_41: 161.64627027511597
    # basic_time_test_de_mons_42: 2068.6068625450134
    # basic_time_test_de_mons_43: 161.38730144500732

    # PHYSIO
    # basic_time_test_physio_0: 1189.2667751312256
    # basic_time_test_physio_1: 1019.8352560997009
    # basic_time_test_physio_2: 1031.8725838661194
    # basic_time_test_physio_3: 3447.3305072784424
    # basic_time_test_physio_4: 1297.86128115654
    # basic_time_test_physio_5: 9225.565055131912
    # basic_time_test_physio_6: 1377.0041360855103
    # basic_time_test_physio_7: err
    # basic_time_test_physio_8: 1245.699406862259
    # basic_time_test_physio_9: 1476.5621535778046
    # basic_time_test_physio_10: 3683.527688264847
    # basic_time_test_physio_11: 3820.8386425971985
    # basic_time_test_physio_12: 4011.121281147003
    # basic_time_test_physio_13: 3810.3192677497864
    # basic_time_test_physio_14: too long
    # basic_time_test_physio_15: too long
    # basic_time_test_physio_16: too long
    # basic_time_test_physio_17: too long
    # basic_time_test_physio_18: 1718.0101916790009
    # basic_time_test_physio_19: 1536.8954167366028
    # basic_time_test_physio_20: 4183.404004812241
    # basic_time_test_physio_21: 3942.4722266197205
    # basic_time_test_physio_22: 3644.452351331711
    # basic_time_test_physio_23: too long
    # basic_time_test_physio_24: too long
    # basic_time_test_physio_25: too long
    # basic_time_test_physio_26: too long
    # basic_time_test_physio_27: 2639.911137342453
    # basic_time_test_physio_28: 2664.324657678604
    # basic_time_test_physio_29: 2703.1563200950623
    # basic_time_test_physio_30: 1335.1774685382843
    # basic_time_test_physio_31: 1327.127035856247
    # basic_time_test_physio_32: 1348.7438204288483
    # basic_time_test_physio_33: 1346.9565255641937
    # basic_time_test_physio_34: 1370.7704150676727
    # basic_time_test_physio_35: 1392.9937829971313
    # basic_time_test_physio_36: 1475.9210562705994
    # basic_time_test_physio_37: 1469.4200489521027
    # basic_time_test_physio_38: 2083.2481853961945
    # basic_time_test_physio_39: 2130.488794565201
    # basic_time_test_physio_40: 1342.9916791915894
    # basic_time_test_physio_41: 1500.8176662921906
    # basic_time_test_physio_42:: too long
    # basic_time_test_physio_43: 1529.4244935512543

