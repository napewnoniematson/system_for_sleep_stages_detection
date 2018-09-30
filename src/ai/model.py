import tensorflow as tf
from src.utils.util import *


class Model:
    def __init__(self, features_train, labels_train, has_normalization=False):
        self.features = features_train
        self.labels = labels_train
        self.__normalize(is_on=has_normalization)
        self.__make()
        self.__compile()
        self.__train()

    def __normalize(self, is_on=True):
        if is_on:
            self.__normalize_on()

    def __normalize_on(self):
        self.features = tf.keras.utils.normalize(self.features, axis=1)

    def __make(self):
        features_amount = len(self.features[0])
        input_shape = (features_amount,)

        self.model = tf.keras.models.Sequential()
        self.model.add(tf.keras.layers.Flatten(input_shape=input_shape))
        self.model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
        self.model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
        self.model.add(tf.keras.layers.Dense(5, activation=tf.nn.softmax))

    def __compile(self):
        self.model.compile(optimizer='adam',
                           loss='sparse_categorical_crossentropy',
                           metrics=['accuracy'])

    def __train(self):
        self.model.fit(self.features, self.labels, epochs=15)

    def save(self):
        self.model.save(MODEL_FILE)

    def load(self):
        self.model = tf.keras.models.load_model(MODEL_FILE)
