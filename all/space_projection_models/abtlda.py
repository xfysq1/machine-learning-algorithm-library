import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import preprocessing

class abt_lda:
    def __init__(self, x, n_components, preprocess=True):
        self.x = x
        self.preprocess = preprocess
        self.n_components = n_components

        if self.preprocess:
            self.Xscaler = preprocessing.StandardScaler().fit(self.x)
            self.x = self.Xscaler.transform(self.x)

    def construct_lda_model(self):
        self.lda = LinearDiscriminantAnalysis(self.n_components)
        self.lda.fit(self.x)

    def extract_lda_feature(self, x_test):
        if self.preprocess:
            x_test = self.Xscaler.transform(x_test)
        return self.lda.transform(x_test)

    def extract_lda_ratio(self):

        a = self.lda.explained_variance_ratio_

        return a
