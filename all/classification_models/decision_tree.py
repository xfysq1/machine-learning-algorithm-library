# -*- coding: utf-8 -*-

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn import metrics

class DecisionTree:
    def __init__(self, x, y, maxdep=None, minspl=2, minlea=1, class_weight=False):

        self.x = x
        self.y = y
        self.maxdep = maxdep
        self.minspl = minspl
        self.minlea = minlea
        self.class_weight = class_weight

        if self.class_weight:
            self.class_weight = 'balanced'
        else:
            self.class_weight = None

    def construct_dt_model(self):

        self.dt = DecisionTreeClassifier(max_depth=self.maxdep, min_samples_split=self.minspl,
                                         min_samples_leaf=self.minlea, class_weight=self.class_weight)
        self.dt.fit(self.x, self.y)

    def extract_dt_samples(self, x_test):

        return self.dt.predict(x_test)

    def tree_dt_model(self, plt):

        plot_tree(self.dt, ax=plt)

    def clfReport(self):

        x_train, x_test, y_train, y_test = train_test_split(self.x, self.y, test_size=0.3)
        self.dt.fit(x_train, y_train)
        pre = self.dt.predict(x_test)

        return metrics.classification_report(y_test, pre)  # 1d array-like

