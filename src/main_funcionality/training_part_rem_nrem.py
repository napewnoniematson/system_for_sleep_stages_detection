import numpy as np
import src.utils.figure as fig
import src.utils.timer as u_timer
import src.ai.model as ann_m
import src.ai.classifier as ann_c
import src.file_system.file_manager as file_m
import src.utils.diagnostic_reliability as dr
from src.file_system.directory_manager import *
from src.model.ann_model_config import ANNModelConfig

default_model_config = ANNModelConfig()
main_path = '/home/pc/mgr/features/features_eog_non_normalized/window_200/physio_window_200/'
norm = True


def training(a):
    """
    case with existing calculated features
    :return:
    """
    # LOAD PATHS
    features_folder_path = main_path + '{}/'
    training_folder_path = features_folder_path + 'training/'
    testing_folder_path = features_folder_path + 'test/'

    # start time
    registered_time = u_timer.get_time()
    # create directories
    create_data_directory(registered_time)
    create_predictions_directory(registered_time)

    # FEATURES FOLDERS
    features_folders = sorted(os.listdir(main_path))
    features_folders = [
        features_folders[a]
        #     features_folders[b]
    ]
    print(features_folders)

    # READ TRAINING FEATURES
    training_features = []
    for ff in features_folders:
        fp = training_folder_path.format(ff)
        temp = []
        for file in os.listdir(fp):
            with open(fp + file, 'r') as f:
                temp.extend(f.readlines())
        training_features.append(temp)
    hypnogram0 = np.asarray([int(f[0]) for f in training_features[0]])

    temp_h, temp_t = [], []
    for h, t in zip(hypnogram0, training_features[0]):
        if h == AWAKE:
            continue
        temp_h.append(int(h) - 1)
        temp_t.append(t)
    hypnogram0 = np.asarray(temp_h)
    training_features = [temp_t]

    # temp = []
    # for feature in training_features:
    #     temp2 = []
    #     for f in feature:
    #         try:
    #             temp2.append(float(f[2:-2]))
    #         except ValueError:
    #             pass
    #     temp.append(temp2)
    # training_features = np.asarray(temp)
    training_features = np.asarray([[float(f[2:-2]) for f in feature] for feature in training_features])

    # NORMALIZE TRAINING FEATURES
    mins, maxes = [], []
    for i in range(len(features_folders)):
        mins.append(training_features[i].min())
        maxes.append(training_features[i].max())
        training_features[i] = (training_features[i] - mins[i]) / (maxes[i] - mins[i]) if norm else training_features[i]

    training_features = training_features.transpose()

    model_config = default_model_config
    input_amount = len(training_features[0])
    output_amount = 2
    model_config.input_amount = input_amount
    model_config.output_amount = output_amount
    model = ann_m.Model(model_config)
    model.train(training_features, hypnogram0)
    cls = ann_c.Classifier(model)

    # READ TEST FEATURES
    evaluates, diagnostics = [], []
    fp = testing_folder_path.format(features_folders[0])
    files = os.listdir(fp)
    for file in files:
        test_features = []
        for ff in features_folders:
            fp = testing_folder_path.format(ff)
            with open(fp + file, 'r') as f:
                test_features.append(f.readlines())
        hypnogram1 = np.asarray([int(f[0]) for f in test_features[0]])

        temp_h, temp_t = [], []
        for h, t in zip(hypnogram1, test_features[0]):
            if h == AWAKE:
                continue
            temp_h.append(int(h) - 1)
            temp_t.append(t)
        hypnogram1 = np.asarray(temp_h)
        test_features = [temp_t]

        test_features = np.asarray([[float(f[2:-2]) for f in feature] for feature in test_features])
        test_features[i] = (test_features[i] - mins[i]) / (maxes[i] - mins[i]) if norm else test_features[i]
        test_features = test_features.transpose()
        evaluate, diagnostic, predictions_arr = cls.evaluate12(test_features, hypnogram1, model_config.output_amount)
        evaluates.append(evaluate)
        diagnostics.append(diagnostic)
        file_m.save(PREDICTIONS_FILE_DIR.format(registered_time, file), [predictions_arr])

    # save model
    model.save(registered_time)
    # save confusion matrix files
    eval_ctr = 0
    for ev, file in zip(evaluates, files):
        cm = np.asarray(ev['matrix'])
        fig.save_to_file_confusion_matrix(cm, ['REM', 'NREM'],
                                          PLOT_IMAGE_FILE_DIR.format(registered_time, eval_ctr, file),
                                          normalize=False).clf()
        fig.save_to_file_confusion_matrix(cm, ['REM', 'NREM'],
                                          NORMALIZED_PLOT_IMAGE_FILE_DIR.format(registered_time, eval_ctr, file),
                                          normalize=True).clf()
        eval_ctr = eval_ctr + 1

    total_tp, total_tn, total_fn, total_fp = 0, 0, 0, 0
    for diag in diagnostics:
        total_tp = total_tp + diag['true_positive']
        total_tn = total_tn + diag['true_negative']
        total_fn = total_fn + diag['false_negative']
        total_fp = total_fp + diag['false_positive']

    total_accuracy = dr.accuracy(total_tp, total_fp, total_fn, total_tn)
    total_sensitivity = dr.sensitivity(total_tp, total_fn)
    total_specificity = dr.specificity(total_fp, total_tn)
    total_ppv = dr.positive_predictive_value(total_tp, total_fp)
    total_npv = dr.negative_predictive_value(total_fn, total_tn)
    result = {
        'features_callbacks': features_folders,
        'model_config': model_config,
        'total_diagnostic': {
            'total_accuracy': total_accuracy,
            'total_sensitivity': total_sensitivity,
            'total_specificity': total_specificity,
            'total_ppv': total_ppv,
            'total_npv': total_npv
        },
        'diagnostic': diagnostics,
        'evaluates': evaluates
    }
    # save results to file
    file_m.save(RESULT_FILE_DIR.format(registered_time), result.items())
    return result
