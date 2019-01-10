import numpy as np
import tensorflow as tf
import src.logger.logger as logger


class Classifier:

    def __init__(self, features_train, labels_train, k):
        self._features_train = features_train
        self._labels_train = labels_train
        self._k = k
        self._prediction = None
        self.features_train_ph = None
        self.labels_train_ph = None
        self.features_test_ph = None
        self._labels_to_matrix()
        self._model()
        self._train()

    def _labels_to_matrix(self):
        self._labels_train = np.eye(len(set(self._labels_train)))[self._labels_train]

    def _model(self):
        feature_number = len(self._features_train[0])
        label_number = len(self._labels_train[0])
        self.features_train_ph = tf.placeholder(shape=[None, feature_number], dtype=tf.float32)
        self.labels_train_ph = tf.placeholder(shape=[None, label_number], dtype=tf.float32)
        self.features_test_ph = tf.placeholder(shape=[None, feature_number], dtype=tf.float32)

    def _distance(self, features_train_ph, features_test_ph):
        return tf.reduce_sum(tf.abs(tf.subtract(features_train_ph, tf.expand_dims(features_test_ph, 1))), axis=2)

    def _train(self):
        logger.info("Training started")
        distance = self._distance(self.features_train_ph, self.features_test_ph)
        _, top_k_indices = tf.nn.top_k(tf.negative(distance), k=self._k)
        top_k_label = tf.gather(self.labels_train_ph, top_k_indices)

        sum_up_predictions = tf.reduce_sum(top_k_label, axis=1)
        self.prediction = tf.argmax(sum_up_predictions, axis=1)

    def classify(self, features_test):
        logger.info("Classify started")
        sess = tf.Session()
        prediction_outcome = sess.run(
            self.prediction,
            feed_dict={
                self.features_train_ph: self._features_train,
                self.features_test_ph: features_test,
                self.labels_train_ph: self._labels_train

            }
        )
        return prediction_outcome

    def evaluate(self, features_test, labels_test):
        accuracy = 0
        prediction_outcome = self.classify(features_test)
        for pred, actual in zip(prediction_outcome, labels_test):
            # if pred == np.argmax(actual):
            if pred == actual:
                accuracy += 1
        return accuracy/len(prediction_outcome)
