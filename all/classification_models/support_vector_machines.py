# -*- coding: utf-8 -*-

from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import metrics

class SupportVectorClassification:
    def __init__(self, x, y, C=1.0, kernel='rbf', class_weight=False, decision='ovr', preprocess=True):

        self.x = x
        self.y = y
        self.C = C
        self.kernel = kernel
        self.class_weight = class_weight
        self.decision = decision
        self.preprocess = preprocess

        if self.class_weight:
            self.class_weight = 'balanced'
        else:
            self.class_weight = None

        if self.preprocess:
            self.Xscaler = preprocessing.StandardScaler().fit(self.x)
            self.x = self.Xscaler.transform(self.x)

    def construct_svc_model(self):

        self.svc = SVC(C=self.C, kernel=self.kernel, class_weight=self.class_weight,
                       decision_function_shape=self.decision)
        self.svc.fit(self.x, self.y)

    def extract_svc_samples(self, x_test):

        if self.preprocess:
            x_test = self.Xscaler.transform(x_test)

        return self.svc.predict(x_test)

    def support_svc_model(self):

        return self.svc.support_

    def clfReport(self):

        x_train, x_test, y_train, y_test = train_test_split(self.x, self.y, test_size=0.3)
        self.svc.fit(x_train, y_train)
        pre = self.svc.predict(x_test)

        return metrics.classification_report(y_test, pre)  # 1d array-like

