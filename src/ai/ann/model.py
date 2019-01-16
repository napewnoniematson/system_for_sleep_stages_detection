import tensorflow as tf
from src.utils.util import *


class Model:
    def __init__(self, input_amount, output_amount, hidden1=32, hidden2=18, has_normalization=False,
                 epochs=EPOCHS_DEFAULT):
        self.input_amount = input_amount
        self.output_amount = output_amount
        if has_normalization:
            self.__normalize()
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

    def train(self, features, labels, epochs):
        self.model.fit(features, labels, epochs=epochs)

    def save(self, file=MODEL_FILE):
        self.model.save(file)

    def load(self):
        self.model = tf.keras.models.load_model(MODEL_FILE)

    @staticmethod
    def loads(file=MODEL_FILE):
        return file, tf.keras.models.load_model(file)
