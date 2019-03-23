import numpy as np
import src.diagnostic_reliability as diagnostic


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
            if p == l:
                accuracy += 1
        tp = matrix[1][1]
        tn = matrix[0][0] + matrix[0][2] + matrix[2][0] + matrix[2][2]
        fn = matrix[1][0] + matrix[1][2]
        fp = matrix[0][1] + matrix[2][1]
        return {
                   "accuracy": accuracy / len(labels_test),
                   "matrix": matrix,
               }, {
                   'true_positive': tp,
                   'true_negative': tn,
                   'false_positive': fp,
                   'false_negative': fn,
                   'accuracy': diagnostic.accuracy(tp, fp, fn, tn),
                   'sensitivity': diagnostic.sensitivity(tp, fn),
                   'specificity': diagnostic.specificity(fp, tn),
                   'positive_predictive_value': diagnostic.positive_predictive_value(tp, fp),
                   'negative_predictive_value': diagnostic.negative_predictive_value(fn, tn),
               }, predictions

    def classify(self, features):
        predictions = self.model.predict([features])
        return [np.argmax(p) for p in predictions]
