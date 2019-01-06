import tensorflow as tf
from src.utils.util import *
from src.model.stage_mode import StageMode
from src.decorator.timer import timer


class Model:
    def __init__(self, features_train, labels_train, has_normalization=False, epochs=EPOCHS_DEFAULT,
                 stage_mode=STAGE_MODE_VALUE_DEFAULT):
        self.features = features_train
        self.labels = labels_train
        self.__normalize(has_normalization=has_normalization)
        self.__make(stage_mode=stage_mode)
        self.__compile(stage_mode=stage_mode)
        self.__train(epochs=epochs)

    def __normalize(self, has_normalization=True):
        if has_normalization:
            self.features = tf.keras.utils.normalize(self.features, axis=1)

    def __make(self, stage_mode):
        features_amount = len(self.features[0])
        input_shape = (features_amount,)
        class_amount = 2 if StageMode.REM_VS_NREM.value == stage_mode else 5
        self.model = tf.keras.models.Sequential()
        self.model.add(tf.keras.layers.Flatten(input_shape=input_shape))
        self.model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
        self.model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
        self.model.add(tf.keras.layers.Dense(class_amount, activation=tf.nn.softmax))

    @staticmethod
    def __loss_calculate_method(stage_mode):
        return 'binary_crossentropy' if StageMode.REM_VS_NREM == stage_mode else 'sparse_categorical_crossentropy'

    def __compile(self, stage_mode):
        self.model.compile(optimizer='adam',
                           loss=self.__loss_calculate_method(stage_mode=stage_mode),
                           metrics=['accuracy'])

    @timer
    def __train(self, epochs=15):
        self.model.fit(self.features, self.labels, epochs)

    def save(self, file=MODEL_FILE):
        self.model.save(file)

    def load(self):
        self.model = tf.keras.models.load_model(MODEL_FILE)

    @staticmethod
    def loads(file=MODEL_FILE):
        return file, tf.keras.models.load_model(file)
