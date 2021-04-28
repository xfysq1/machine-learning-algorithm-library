import math
import numpy as np
from sklearn import preprocessing
from sklearn.linear_model import Lasso
import pandas as pd


class fs_LASSO:

    def __init__(self, z, n_components, preprocess=True):

        self.z = z
        self.preprocess = preprocess
        self.n_components = n_components

        if self.preprocess:
            self.Xscaler = preprocessing.StandardScaler().fit(self.z)
            self.z = self.Xscaler.transform(self.z)

    def construct_LASSO_model(self):
        p1 = []
        dataSet = self.z
        n, m = np.shape(dataSet)  # 获取数据集行数和列数
        x = [0] * n  # 初始化特征x和类别y向量
        y = [0] * n
        for i in range(n):  # 得到标签向量
            y[i] = dataSet[i][m - 1]
        for j in range(m - 1):  # 获取每个特征的向量
            for k in range(n):
                x[k] = dataSet[k][j]
            x = np.array(x).reshape(n,1)     #数据重构为2维数组
            p1.append(Lasso(alpha = 0.01).fit(x, y).coef_)     #计算LASSO相关系数

        p1 = np.array(p1)
        p1 = np.fabs(p1)
        LASSO_index = np.argsort(-p1)
        LASSO_index = LASSO_index[0:self.n_components]
        p1 = sorted(p1, reverse=True)     #降序排列
        p1 = p1[0:self.n_components]     #提取前n_components个系数
        p1 = np.array(p1)     #转换为2维数组
        p1 = np.ravel(p1)     #降为1维数组

        return p1,LASSO_index

    def extract_LASSO_feature(self, x_test):
        p2 = []
        x_test = self.Xscaler.transform(x_test)
        n, m = np.shape(x_test)  # 获取数据集行数和列数
        x = [0] * n  # 初始化特征x和类别y向量
        y = [0] * n
        for i in range(n):  # 得到标签向量
            y[i] = x_test[i][m - 1]
        for j in range(m - 1):  # 获取每个特征的向量
            for k in range(n):
                x[k] = x_test[k][j]
            x = np.array(x).reshape(n,1)     #数据重构为2维数组
            p2.append(Lasso(alpha = 0.01).fit(x, y).coef_)     #计算LASSO相关系数

        p2 = np.array(p2)
        p2 = np.fabs(p2)
        p2 = np.ravel(p2)
        LASSO_index = np.argsort(-p2)
        LASSO_index = LASSO_index[0:self.n_components]
        LASSO_index = np.array(LASSO_index).reshape(self.n_components,1)
        p2 = sorted(p2, reverse=True)     #降序排列
        p2 = p2[0:self.n_components]     #提取前n_components个系数
        p2 = np.array(p2).reshape(self.n_components, 1)
        p = np.hstack((p2,LASSO_index))
        return p



