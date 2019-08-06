import tensorflow as tf
from src.utils.util import *


class Model:
    def __init__(self, model_config):
        self.model_config = model_config
        self.__make()
        self.__compile()

    def __make(self):
        input_shape = (self.model_config.input_amount,)
        class_amount = self.model_config.output_amount  # AWAKE, REM+STAGE1, STAGE2+STAGE3

        self.model = tf.keras.models.Sequential()
        self.model.add(tf.keras.layers.Flatten(input_shape=input_shape))
        self.model.add(tf.keras.layers.Dense(self.model_config.hidden1, activation=self.model_config.activation_h1))
        self.model.add(tf.keras.layers.Dense(self.model_config.hidden2, activation=self.model_config.activation_h2))
        self.model.add(tf.keras.layers.Dense(class_amount, activation=self.model_config.activation_output))

    def __compile(self):
        self.model.compile(optimizer=self.model_config.optimizer,
                           loss=self.model_config.loss,
                           metrics=self.model_config.metrics)

    def train(self, features, labels):
        self.model.fit(features, labels, epochs=self.model_config.epochs, batch_size=self.model_config.batch_size)

    def save(self, date):
        self.model.save(MODEL_FILE_DIR.format(date))

    def load(self):
        self.model = tf.keras.models.load_model()

    @staticmethod
    def loads(file):
        return file, tf.keras.models.load_model(file)
