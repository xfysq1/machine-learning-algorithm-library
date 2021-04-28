# -*- coding: utf-8 -*-

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

class RandomForest:
    def __init__(self, x, y, ntree, nfea, weight):
        self.x = x
        self.y = y
        self.ntree = ntree
        self.nfea = nfea
        self.weight = weight

        if self.weight == 'None':
            self.weight = None

    def construct_rf_model(self):

        self.rf = RandomForestClassifier(n_estimators=self.ntree, max_features=self.nfea,
                                         class_weight=self.weight)
        self.rf.fit(self.x, self.y)

    def extract_rf_samples(self, x_test):

        return self.rf.predict(x_test)

    def feaimportance_rf_model(self):

        return self.rf.feature_importances_

    def clfReport(self):

        x_train, x_test, y_train, y_test = train_test_split(self.x, self.y, test_size=0.3)
        self.rf.fit(x_train, y_train)
        pre = self.rf.predict(x_test)

        return metrics.classification_report(y_test, pre)  # 1d array-like


