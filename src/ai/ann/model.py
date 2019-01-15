import tensorflow as tf
from src.utils.util import *


class Model:
    def __init__(self, features_train, labels_train, hidden1=32, hidden2=18, has_normalization=False,
                 epochs=EPOCHS_DEFAULT):
        self.features = features_train
        self.labels = labels_train
        if has_normalization:
            self.__normalize()
        self.__make(hidden1=hidden1, hidden2=hidden2)
        self.__compile()
        # self.__train(epochs=epochs)

    def __normalize(self):
        self.features = tf.keras.utils.normalize(self.features, axis=1)

    def __make(self, hidden1, hidden2):
        features_amount = len(self.features[0])
        input_shape = (features_amount,)
        class_amount = 3  # AWAKE, REM+STAGE1, STAGE2+STAGE3

        self.model = tf.keras.models.Sequential()
        self.model.add(tf.keras.layers.Flatten(input_shape=input_shape))
        self.model.add(tf.keras.layers.Dense(hidden1, activation=tf.nn.relu))
        self.model.add(tf.keras.layers.Dense(hidden2, activation=tf.nn.relu))
        self.model.add(tf.keras.layers.Dense(class_amount, activation=tf.nn.softmax))

    def __compile(self):
        self.model.compile(optimizer='adam',
                           loss='sparse_categorical_crossentropy',
                           metrics=['accuracy'])

    def train(self, f, l, epochs, batch_size=32):
        self.model.fit(x=f, y=l, epochs=epochs, batch_size=batch_size)

    def save(self, file=MODEL_FILE):
        self.model.save(file)

    def load(self):
        self.model = tf.keras.models.load_model(MODEL_FILE)

    @staticmethod
    def loads(file=MODEL_FILE):
        return file, tf.keras.models.load_model(file)
