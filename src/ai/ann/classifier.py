import numpy as np


class Classifier:
    def __init__(self, model):
        self.model = model.model

    def evaluate(self, features_test, labels_test):
        val_loss, val_acc = self.model.evaluate(features_test, labels_test)
        return {
            "loss": val_loss,
            "accuracy": val_acc
        }

    def evaluate2(self, features_test, labels_test):
        accuracy = 0
        predictions = self.classify(features_test)
        for p, l in zip(predictions, labels_test):
            # print("p = %d l = %d" % (p,l))
            if p == l:
                accuracy += 1
        return accuracy / len(labels_test)

    def classify(self, features):
        predictions = self.model.predict([features])
        return [np.argmax(p) for p in predictions]
