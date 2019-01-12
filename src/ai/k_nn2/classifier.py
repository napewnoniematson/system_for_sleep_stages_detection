from sklearn.neighbors import KNeighborsClassifier


class Classifier:

    def __init__(self, features_train, labels_train, k):
        self.features_train = features_train
        self.labels_train = labels_train
        self.k = k
        self.__model()
        self.__train()

    def __model(self):
        self.neigh = KNeighborsClassifier(n_neighbors=self.k)

    def __train(self):
        self.neigh.fit(self.features_train, self.labels_train)

    def classify(self, features_test):
        return self.neigh.predict(features_test)

    def evaluate(self, features_test, labels_test):
        accuracy = 0
        prediction_outcome = self.classify(features_test)
        for pred, actual in zip(prediction_outcome, labels_test):
            if pred == actual:
                accuracy += 1
        return accuracy / len(prediction_outcome)
