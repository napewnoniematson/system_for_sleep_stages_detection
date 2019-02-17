import tensorflow as tf
from src.utils.util import *


class Model:
    def __init__(self, input_amount, output_amount, hidden1=32, hidden2=18):
        self.input_amount = input_amount
        self.output_amount = output_amount
        self.__make(hidden1=hidden1, hidden2=hidden2)
        self.__compile()
        # self.__train(epochs=epochs)

    def __normalize(self):
        self.features = tf.keras.utils.normalize(self.features, axis=1)

    def __make(self, hidden1, hidden2):
        input_shape = (self.input_amount,)
        class_amount = self.output_amount  # AWAKE, REM+STAGE1, STAGE2+STAGE3

        self.model = tf.keras.models.Sequential()
        self.model.add(tf.keras.layers.Flatten(input_shape=input_shape))
        self.model.add(tf.keras.layers.Dense(hidden1, activation=tf.nn.relu))
        self.model.add(tf.keras.layers.Dense(hidden2, activation=tf.nn.relu))
        self.model.add(tf.keras.layers.Dense(class_amount, activation=tf.nn.softmax))

    def __compile(self):
        self.model.compile(optimizer='adam',
                           loss='sparse_categorical_crossentropy',
                           metrics=['accuracy'])

    def train(self, features, labels, epochs, batch_size):
        self.model.fit(features, labels, epochs=epochs, batch_size=batch_size)

    def save(self, date):
        self.model.save(MODEL_FILE_DIR.format(date))

    def load(self):
        self.model = tf.keras.models.load_model()

    @staticmethod
    def loads(file):
        return file, tf.keras.models.load_model(file)
