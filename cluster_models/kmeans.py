# -*- coding: utf-8 -*-

from sklearn.cluster import KMeans
from sklearn import preprocessing

class KMEANS:
    """
    This module is to construct a k-means model for cluster analysis.

    Parameters
    ----------
    x (n_samples, n_features) – The training input samples
    n_clusters – The number of clusters to form as well as the number of centroids to generate
    preprocess (default = True) - the preprocessing of the data

    Attributes
    ----------
    kmeans - model of KMeans

    """
    def __init__(self, x, n_clusters, preprocess = True):

        self.x = x
        self.n_clusters = n_clusters
        self.preprocess = preprocess

        if self.preprocess:  # 对训练数据预处理
            self.Xscaler = preprocessing.StandardScaler().fit(self.x)
            self.x = self.Xscaler.transform(self.x)

    def construct_kmeans_model(self):
        """
        Function to construct a kmeans model.

        """
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=0).fit(self.x)

    def extract_kmeans_samples(self, x_test):
        """
        Function to extract the kmeans samples of given data using the trained-well kmeans model.

        Parameters
        ----------
        x_test (n_samples, n_features) - The testing samples

        """
        if self.preprocess:
            x_test = self.Xscaler.transform(x_test)

        return self.kmeans.predict(x_test)

