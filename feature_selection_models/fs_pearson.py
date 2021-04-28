# -*- coding: utf-8 -*-
import math
import numpy as np
from sklearn import preprocessing
import pandas as pd
import time
'''
time_start=time.time()
time_end=time.time()
print('time cost',time_end-time_start,'s')
'''
class fs_pearson:

    def __init__(self, z, n_components, preprocess=True):

        self.z = z
        self.preprocess = preprocess
        self.n_components = n_components

        if self.preprocess:
            self.Xscaler = preprocessing.StandardScaler().fit(self.z)
            self.z = self.Xscaler.transform(self.z)

    # 计算特征和类的平均值
    def calcMean(x,y):
        sum_x = sum(x)
        sum_y = sum(y)
        n = len(x)
        x_mean = float(sum_x + 0.0) / n
        y_mean = float(sum_y + 0.0) / n
        return x_mean, y_mean

    # 计算Pearson系数
    def calcPearson(x,y):
        x_mean, y_mean = fs_pearson.calcMean(x,y)  # 计算x,y向量平均值
        n = len(x)
        sumTop = 0.0
        sumBottom = 0.0
        x_pow = 0.0
        y_pow = 0.0
        for i in range(n):
            sumTop += (x[i] - x_mean) * (y[i] - y_mean)
        for i in range(n):
            x_pow += math.pow(x[i] - x_mean, 2)
        for i in range(n):
            y_pow += math.pow(y[i] - y_mean, 2)
        sumBottom = math.sqrt(x_pow * y_pow)
        p = sumTop / sumBottom
        p = abs(p)
        return p

    # 计算每个特征的spearman系数，返回数组
    def construct_pearson_model(self):
        score = []
        dataSet = self.z
#        dataSet = dataSet.values.tolist()     #转换为列表
        n , m = np.shape(dataSet)  # 获取数据集行数和列数
        x = [0]*n  # 初始化特征x和类别y向量
        y = [0]*n
        for i in range(n):  # 得到标签向量
            y[i] = dataSet[i][m-1]
        for j in range(m - 1):  # 获取每个特征的向量
            for k in range(n):
                x[k] = dataSet[k][j]
            score.append(fs_pearson.calcPearson(x,y))     #计算pearson系数

        score = np.array(score)
        score = np.fabs(score)     #对计算得到的系数取绝对值
        pearson_index = np.argsort(-score)     #获取降序排列的索引
        pearson_index = pearson_index[0:self.n_components]     #提取前n_components个索引
        score = sorted(score,reverse=True)     #降序排列
        score = score[0:self.n_components]     #提取前n_components个系数
#        score = np.array(score).reshape(self.n_components,1)     #转换为2维数组
        return score,pearson_index

    def extract_pearson_feature(self, x_test):
        score = []
#        x_test = x_test.values.tolist()     #转换为列表
        x_test = self.Xscaler.transform(x_test)
        n, m = np.shape(x_test)  # 获取数据集行数和列数
        x = [0] * n  # 初始化特征x和类别y向量
        y = [0] * n
        for i in range(n):  # 得到标签向量
            y[i] = x_test[i][m - 1]
        for j in range(m - 1):  # 获取每个特征的向量
            for k in range(n):
                x[k] = x_test[k][j]
            score.append(fs_pearson.calcPearson(x, y))     #计算pearson系数

        score = np.array(score)
        score = np.fabs(score)
        pearson_index = np.argsort(-score)
        pearson_index = pearson_index[0:self.n_components]
        pearson_index = np.array(pearson_index).reshape(self.n_components,1)     #转换为2维数组
        score = sorted(score, reverse=True)     #降序排列
        score = score[0:self.n_components]     #提取前n_components个系数
        score = np.array(score).reshape(self.n_components,1)     #转换为2维数组
        score_index = np.hstack((score,pearson_index))     #水平合并

        return score_index
