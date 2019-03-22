# SLEEP STAGE
UNKNOWN = -1
AWAKE = 0
REM = 1
NON_REM = 2
MOVEMENT_TIME = 3

# RHYTHM FREQ
ALPHA_LOW = 8
ALPHA_HIGH = 13
BETA_LOW = 13
BETA_HIGH = 20
THETA_LOW = 4
THETA_HIGH = 8
DELTA_LOW = 0.5
DELTA_HIGH = 4

# DEFAULTS

WINDOW_DEFAULT = 5
EPOCHS_DEFAULT = 5
MODEL_NAME = 'model_{}'
DEPRECATED_REASON = 'Old issue solution'

# DEFAULTS FOR FUNCTIONS
SAMPLING_FREQ_DEFAULT = 100
BAND_DEFAULT = [0.5, 4, 7, 12, 30]  # which are delta, theta, alpha, beta
EMBEDDING_DIMENSION_DEFAULT = 2
EMBEDDING_LAG_DEFAULT = 4
MEAN_PERIOD_DEFAULT = SAMPLING_FREQ_DEFAULT * 1
DIFF_SEQ_DEFAULT = None
KMAX_DEFAULT = 1
AVERAGE_VALUE_DEFAULT = None
BOX_SIZE_LIST_DEFAULT = None
SINGULAR_VALUES_DEFAULT = None
PERMUTATION_ORDER_DEFAULT = 5
POWER_RATIO_DEFAULT = None

# DICTIONARY_KEYS
SAMPLING_FREQ = 'sampling_freq'
BAND = 'band'
EMBEDDING_DIMENSION = 'embedding_dimension'
EMBEDDING_LAG = 'embedding_lag'
MEAN_PERIOD = 'mean_period'
DIFF_SEQ = 'differential_sequence'
KMAX = 'kmax'
AVERAGE_VALUE = 'average_value'
BOX_SIZE_LIST = 'box_size_list'
SINGULAR_VALUES = 'singular_values'
PERMUTATION_ORDER = 'permutation_order'
POWER_RATIO = 'power_ratio'


PHYSIONET_DIR = '/home/pc/mgr/physionet'
PHYSIONET_DIR_LOAD = PHYSIONET_DIR + '/{}'

DE_MONS_DIR = '/home/pc/mgr/de_mons'
DE_MONS_DIR_LOAD = DE_MONS_DIR + '/{}'

LOG_DATE_TIME_FORMAT = '%Y-%m-%d %H:%M:%S'

# DIRECTORY
MAIN_MODEL_DIR = '/home/pc/mgr/model'
MAIN_FEATURE_DIR = '/home/pc/mgr/features'
DATE_FEATURE_DIR = MAIN_FEATURE_DIR + '/{}'
TEST_FEATURE_DIR = DATE_FEATURE_DIR + '/test'
TRAINING_FEATURE_DIR = DATE_FEATURE_DIR + '/training'
PREDICTIONS_DIR = DATE_FEATURE_DIR + '/predictions'
# FILE
TEST_FEATURE_FILE_DIR = TEST_FEATURE_DIR + '/{}'
TRAINING_FEATURE_FILE_DIR = TRAINING_FEATURE_DIR + '/{}'
PREDICTIONS_FILE_DIR = PREDICTIONS_DIR + '/{}'
RESULT_FILE_DIR = DATE_FEATURE_DIR + '/result'
PLOT_IMAGE_FILE_DIR = DATE_FEATURE_DIR + '/cm{}.png'
NORMALIZED_PLOT_IMAGE_FILE_DIR = DATE_FEATURE_DIR + '/cm{}_norm.png'
MODEL_FILE_DIR = MAIN_MODEL_DIR + '/{}.model'
