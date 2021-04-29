# -*- coding: utf-8 -*-

from sklearn.naive_bayes import GaussianNB, MultinomialNB, ComplementNB, BernoulliNB
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import metrics

class NaiveBayesian:
    def __init__(self, x, y, type='GaussianNB', a=1.0, bina=0.0, preprocess=True):

        self.x = x
        self.y = y
        self.type = type
        self.a = a
        self.bina = bina
        self.preprocess = preprocess


    def construct_nbc_model(self):

        if (self.type=='GaussianNB'):
            self.nbc = GaussianNB()
        elif (self.type=='MultinomialNB'):
            self.nbc = MultinomialNB(alpha=self.a)
        elif (self.type=='ComplementNB'):
            self.nbc = ComplementNB(alpha=self.a)
        else:
            self.nbc = BernoulliNB(alpha=self.a, binarize=self.bina)
            if self.preprocess:
                self.Xscaler = preprocessing.StandardScaler().fit(self.x)
                self.x = self.Xscaler.transform(self.x)

        self.nbc.fit(self.x, self.y)

    def extract_nbc_samples(self, x_test):

        if (self.type=='BernoulliNB'):
            if self.preprocess:
                x_test = self.Xscaler.transform(x_test)

        return self.nbc.predict(x_test)

    def clfReport(self):
        x_train, x_test, y_train, y_test = train_test_split(self.x, self.y, test_size=0.3)
        self.nbc.fit(x_train, y_train)
        pre = self.nbc.predict(x_test)

        return metrics.classification_report(y_test, pre)  # 1d array-like


