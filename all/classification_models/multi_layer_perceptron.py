# -*- coding: utf-8 -*-

from sklearn.neural_network import MLPClassifier
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import metrics

class MultiLayerPerceptron:
    def __init__(self, x, y, hidden=(100,), active='relu', solver='adam', a=0.0001):

        self.x = x
        self.y = y
        self.hidden = hidden
        self.active = active
        self.solver = solver
        self.a = a

        self.Xscaler = preprocessing.StandardScaler().fit(self.x)
        self.x = self.Xscaler.transform(self.x)

    def construct_mlp_model(self):

        self.mlp = MLPClassifier(hidden_layer_sizes=self.hidden, activation=self.active,
                                 solver=self.solver, alpha=self.a)
        self.mlp.fit(self.x, self.y)

    def extract_mlp_samples(self, x_test):

        x_test = self.Xscaler.transform(x_test)

        return self.mlp.predict(x_test)

    def clfReport(self):
        x_train, x_test, y_train, y_test = train_test_split(self.x, self.y, test_size=0.3)
        self.mlp.fit(x_train, y_train)
        pre = self.mlp.predict(x_test)

        return metrics.classification_report(y_test, pre)  # 1d array-like

