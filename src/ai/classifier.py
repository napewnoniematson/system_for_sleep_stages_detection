import numpy as np


class Classifier:
    def __init__(self, model):
        self.model = model

    def evaluate(self, features_test, labels_test):
        val_loss, val_acc = self.model.evaluate(features_test, labels_test)
        return {
            "loss": val_loss,
            "accuracy": val_acc
        }

    def verify(self, features):
        predictions = self.model.predict([features])
        return np.argmax(predictions[0])
