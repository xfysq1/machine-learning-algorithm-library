import math
import numpy as np
from sklearn import preprocessing
from sklearn import metrics
import pandas as pd

class fs_MI:

    def __init__(self, z, n_components, preprocess=True):

        self.z = z
        self.preprocess = preprocess
        self.n_components = n_components

        if self.preprocess:
            self.Xscaler = preprocessing.StandardScaler().fit(self.z)
            self.z = self.Xscaler.transform(self.z)

    #标准互信息
    def construct_MI_model(self):
        score = []
        dataSet = self.z
        n , m = np.shape(dataSet)  # 获取数据集行数和列数
        x = [0]*n  # 初始化特征x和类别y向量
        y = [0]*n
        for i in range(n):  # 得到标签向量
            y[i] = dataSet[i][m-1]
        for j in range(m - 1):  # 获取每个特征的向量
            for k in range(n):
                x[k] = dataSet[k][j]
            score.append(metrics.normalized_mutual_info_score(x, y))     #计算MI系数

        score = np.array(score)
        score = np.fabs(score)
        MI_index = np.argsort(-score)
        MI_index = MI_index[0:self.n_components]
        score = sorted(score,reverse=True)     #降序排列
        score = score[0:self.n_components]     #提取前n_components个系数
        return score,MI_index

    def extract_MI_feature(self,x_test):
        score = []
        x_test = self.Xscaler.transform(x_test)
        n , m = np.shape(x_test)  # 获取数据集行数和列数
        x = [0]*n  # 初始化特征x和类别y向量
        y = [0]*n
        for i in range(n):  # 得到标签向量
            y[i] = x_test[i][m-1]
        for j in range(m - 1):  # 获取每个特征的向量
            for k in range(n):
                x[k] = x_test[k][j]
            score.append(metrics.normalized_mutual_info_score(x, y))     #计算MI系数

        score = np.array(score)
        score = np.fabs(score)
        MI_index = np.argsort(-score)
        MI_index = MI_index[0:self.n_components]
        MI_index = np.array(MI_index).reshape(self.n_components,1)     #转换为2维数组
        score = sorted(score,reverse=True)     #降序排列
        score = score[0:self.n_components]     #提取前n_components个系数
        score = np.array(score).reshape(self.n_components,1)     #转换为2维数组
        score_index = np.hstack((score,MI_index))

        return score_index

    '''
    #互信息
    def calc_MI(X, Y, n):
        c_XY = np.histogram2d(X, Y, n)[0]
        c_X = np.histogram(X, n)[0]
        c_Y = np.histogram(Y, n)[0]

        H_X = fs_MI.shan_entropy(c_X)
        H_Y = fs_MI.shan_entropy(c_Y)
        H_XY = fs_MI.shan_entropy(c_XY)

        MI = H_X + H_Y - H_XY
        return MI

    def shan_entropy(c):
        c_normalized = c / float(np.sum(c))
        c_normalized = c_normalized[np.nonzero(c_normalized)]
        H = -sum(c_normalized * np.log2(c_normalized))
        return H
        
    def construct_MI_model(self):
        p1 = []
        dataSet = self.z
        n , m = np.shape(dataSet)  # 获取数据集行数和列数
        x = [0]*n  # 初始化特征x和类别y向量
        y = [0]*n
        for i in range(n):  # 得到类向量
            y[i] = dataSet[i][m-1]
        for j in range(m - 1):  # 获取每个特征的向量，并计算Pearson系数，存入到列表中
            for k in range(n):
                x[k] = dataSet[k][j]
            p1.append(self.calc_MI(x, y,n))
        p1 = sorted(p1,reverse=True)
        p1 = p1[0:self.n_components]
        return p1
    '''


