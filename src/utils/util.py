# DEFAULTS
from src.model.stage_mode import StageMode

WINDOW_WIDTH_DEFAULT = 5
STAGE_MODE_VALUE_DEFAULT = StageMode.REM_VS_NREM.value
EPOCHS_DEFAULT = 15
TRAINING_PART_PERCENTAGE_DEFAULT = 70
SAMPLING_FREQ_DEFAULT = 200
MODEL_NAME = 'model_{}'
MODEL_NAME_DEFAULT = 'model_default_name'

# GUI
TITLE = "System for sleep stages detection"
MODEL_NAME_TEXT = 'Model name'
EXAMINATIONS_TEXT = "Examinations"
FEATURES_TEXT = "Features"
WINDOW_WIDTH_TEXT = "Window width"
STAGE_MODE_TEXT = 'Stage mode'
EPOCHS_TEXT = 'Epochs'
TRAINING_PART_TEXT = 'Training part'
ACCURACY_TEXT = 'Accuracy'
TRAINING_TEXT = "TRAIN"
CLASSIFY_TEXT = 'CLASSIFY'
SAVE_MODEL_TEXT = "Save model"
LOAD_MODEL_TEXT = "Load model"
UPDATE_TEXT = 'Update'
FIGURE_TEXT = 'Figure'
RESULT_TEXT = "Result"
EXAMINATIONS_FOR_CLASSIFY_TEXT = "Examination for classify"
EXAMINATIONS_FOR_CLASSIFY_LENGTH_TEXT = "Examination for classify length"
CLASSIFY_BUTTON_TEXT = "CLASSIFY"
CONTROLLER_NOT_FOUND_EXCEPTION = "View's controller not found"
TRAINING_FINISHED_MESSAGE = 'Training is finished'
LOADING_MODEL_FINISHED_MESSAGE = 'Loading is finished'
SAVING_MODEL_FINISHED_MESSAGE = 'Saving is finished'
UPDATING_RESULT_FINISHED_MESSAGE = 'Updating is finished'

# DICTIONARY_KEYS
EXAMINATIONS_KEY = "examinations"
FEATURES_KEY = "features"
WINDOW_WIDTH_KEY = "window_width"
STAGE_MODE_KEY = 'stage_mode'
EPOCHS_KEY = 'epochs'
TRAINING_PART_KEY = 'training_part_percentage'
CLASSIFY_KEY = 'classify'

FEATURES_TITLES_KEY = "titles"
FEATURES_CALLBACKS_KEY = "callbacks"

SAMPLING_FREQ = 'sampling_freq'

# EXCEPTION MESSAGES
WINDOW_SIZE_EXCEPTION_MESSAGE = "Illegal window size exception"

# PATHS
PHYSIONET_PHYSIOBANK_DATABASE_SLEEP_EDFX = "/home/pc/mgr/mgr_data/SC4002E0-PSG.edf"
PHYSIONET_PHYSIOBANK_DATABASE_SHHPSGDB = "/home/pc/mgr/mgr_data/0000.edf"
UNIVERSITE_DE_MONS = "/home/pc/mgr/mgr_data/{}.edf"
UNIVERSITE_DE_MONS_HYPNOGRAM_AASM = \
    "/home/pc/mgr/mgr_data/HypnogramAASM_{}.txt"
UNIVERSITE_DE_MONS_HYPNOGRAM_RK = \
    "/home/pc/mgr/mgr_data/HypnogramR&K_{}.txt"

# AI
FEATURE_NAME = "eeg_features"
MODEL_FILE_EXTENSION = '.model'
MODEL_FILE = MODEL_NAME_DEFAULT + MODEL_FILE_EXTENSION

LOG_DATE_TIME_FORMAT = '%Y-%m-%d %H:%M:%S'
# ***********************LOAD_EDF************************
# ECG
LOAD_EDF_ECG_NO = 0

# EEG
LOAD_EDF_EEG_FPI_A2_NO = 1
LOAD_EDF_EEG_CZ_A1_NO = 2
LOAD_EDF_EEG_O1_A2_NO = 15
LOAD_EDF_EEG_FP2_A1_NO = 16
LOAD_EDF_EEG_O2_A1_NO = 17
LOAD_EDF_EEG_CZ2_A1_NO = 18

# EMG
LOAD_EDF_EMG1_NO = 3
LOAD_EDF_EMG2_NO = 19
LOAD_EDF_EMG3_NO = 22

# EOG
LOAD_EDF_EOG1_A2_NO = 4
LOAD_EDF_EOG2_A2_NO = 14

# OTHER
LOAD_EDF_VTH_NO = 5
LOAD_EDF_VAB_NO = 6
LOAD_EDF_NAF2P_A1_NO = 7
LOAD_EDF_NAF1_NO = 8
LOAD_EDF_PHONO_NO = 9
LOAD_EDF_PR_NO = 10
LOAD_EDF_SAO2_NO = 11
LOAD_EDF_PCPAP_NO = 12
LOAD_EDF_POS_NO = 13
LOAD_EDF_PULSE_NO = 20
LOAD_EDF_VTOT_NO = 21

# **********LOAD_FROM_EEG_SIGNAL_ARRAY****************

# EEG
EEG_FPI_A2_NO = 0
EEG_CZ_A1_NO = 1
EEG_O1_A2_NO = 2
EEG_FP2_A1_NO = 3
EEG_O2_A1_NO = 4
EEG_CZ2_A1_NO = 5
