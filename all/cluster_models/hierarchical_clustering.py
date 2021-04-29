# -*- coding: utf-8 -*-

from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial import distance
from sklearn import preprocessing

import numpy as np

class HierarchicalClustering:
    def __init__(self, x, t, method='single', criterion='inconsistent', preprocess=True):

        self.x = x
        self.t = t
        self.method = method
        self.criterion = criterion
        self.preprocess = preprocess

        if self.preprocess:  # 对训练数据预处理
            self.Xscaler = preprocessing.StandardScaler().fit(self.x)
            self.x = self.Xscaler.transform(self.x)

    def construct_hcc_model(self):

        dis_x = distance.pdist(self.x, metric='euclidean')
        self.z = linkage(dis_x, method=self.method)
        self.xlabels_hcc = fcluster(self.z, t=self.t, criterion=self.criterion)

        return self.z

    def func_hcc_labels(self):

        return self.xlabels_hcc

    def extract_hcc_samples(self, x_test):

        if self.preprocess:
            x_test = self.Xscaler.transform(x_test)


        n_clus = max(self.xlabels_hcc)
        center_class = [[] for i in range(n_clus)]
        for i in range(n_clus):
            b = []
            for index, nums in enumerate(self.xlabels_hcc):
                if nums == (i+1):
                    b.append(index)
            no_i_class = self.x[b]
            center_class[i].append(np.mean(no_i_class, axis=0))

        dis_class = [[] for i in range(n_clus)]
        for i in range(n_clus):
            dis_class[i] = np.sqrt(np.sum(np.asarray(np.mat(center_class[i]) - x_test)**2, axis=1))

        dis_class = (np.mat(dis_class).T).A
        n_row, n_col = x_test.shape
        test_labels = [[] for i in range(n_row)]
        for i in range(n_row):
            test_labels[i].append(np.argmin(dis_class[i])+1)

        return test_labels




