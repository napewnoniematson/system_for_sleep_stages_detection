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

    def evaluate2(self, features_test, labels_test, class_amount):
        accuracy = 0
        matrix = [[0 for _ in range(class_amount)] for _ in range(class_amount)]
        predictions = self.classify(features_test)
        for p, l in zip(predictions, labels_test):
            matrix[l][p] += 1
            # print("p = %d l = %d" % (p,l))
            if p == l:
                accuracy += 1
        # print(predictions)
        return {
            "accuracy": accuracy / len(labels_test),
            "matrix": matrix
            # "original_hypnogram": labels_test,
            # "classify_hypnogram": predictions
        }, predictions

    def classify(self, features):
        predictions = self.model.predict([features])
        return [np.argmax(p) for p in predictions]
