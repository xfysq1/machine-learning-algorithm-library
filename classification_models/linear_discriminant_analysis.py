# -*- coding: utf-8 -*-

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import metrics

class LinearDiscriminant:
    def __init__(self, x, y, solver='svd', preprocess=True):

        self.x = x
        self.y = y
        self.solver = solver
        self.preprocess = preprocess

        if self.preprocess:
            self.Xscaler = preprocessing.StandardScaler().fit(self.x)
            self.x = self.Xscaler.transform(self.x)

    def construct_lda_model(self):

        self.lda = LinearDiscriminantAnalysis(solver=self.solver)
        self.lda.fit(self.x, self.y)

    def extract_lda_samples(self, x_test):

        if self.preprocess:
            x_test = self.Xscaler.transform(x_test)

        return self.lda.predict(x_test)

    def weight_lda_model(self):

        return self.lda.coef_

    def clfReport(self):

        x_train, x_test, y_train, y_test = train_test_split(self.x, self.y, test_size=0.3)
        self.lda.fit(x_train, y_train)
        pre = self.lda.predict(x_test)

        return metrics.classification_report(y_test, pre)  # 1d array-like

