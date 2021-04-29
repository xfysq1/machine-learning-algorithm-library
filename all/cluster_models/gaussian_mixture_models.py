# -*- coding: utf-8 -*-

from sklearn.mixture import GaussianMixture
from sklearn import preprocessing

class GaussianMixtureModels:
    def __init__(self, x, n_components, covariance_type='full', preprocess=True):

        self.x = x
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.preprocess = preprocess

        if self.preprocess:  # 对训练数据预处理
            self.Xscaler = preprocessing.StandardScaler().fit(self.x)
            self.x = self.Xscaler.transform(self.x)

    def construct_gmm_model(self):

        self.gmm = GaussianMixture(n_components=self.n_components, covariance_type=self.covariance_type, random_state=0)
        self.gmm.fit(self.x)

    def extract_gmm_samples(self, x_test):

        if self.preprocess:
            x_test = self.Xscaler.transform(x_test)

        return self.gmm.predict(x_test)

    def bic_gmm_model(self):

        return self.gmm.bic(self.x)

