import math
import numpy as np
from sklearn import preprocessing
from xgboost import XGBClassifier
import pandas as pd

class fs_XGboost:

    def __init__(self, z, n_components, preprocess=True):

        self.z = z
        self.preprocess = preprocess
        self.n_components = n_components

        if self.preprocess:
            self.Xscaler = preprocessing.StandardScaler().fit(self.z)
            self.z = self.Xscaler.transform(self.z)

    #XGboost
    def construct_XGboost_model(self):
        dataSet = self.z
        n , m = np.shape(dataSet)  # 获取数据集行数和列数
        x = dataSet[:,0:m-1]  # 初始化特征x和类别y向量
        y = dataSet[:,m-1]
        model = XGBClassifier()
        model.fit(x, y)
        score = model.feature_importances_     #计算XGboost系数

        score = np.array(score)
        score = np.fabs(score)
        XGboost_index = np.argsort(-score)
        XGboost_index = XGboost_index[0:self.n_components]
        score = sorted(score,reverse=True)     #降序排列
        score = score[0:self.n_components]     #提取前n_components个系数
        return score,XGboost_index

    def extract_XGboost_feature(self,x_test):
        x_test = self.Xscaler.transform(x_test)
        n , m = np.shape(x_test)  # 获取数据集行数和列数
        x = x_test[:,0:m-1]  # 初始化特征x和类别y向量
        y = x_test[:,m-1]
        model = XGBClassifier()
        model.fit(x, y)
        score = model.feature_importances_     #计算XGboost系数

        score = np.array(score)
        score = np.fabs(score)
        XGboost_index = np.argsort(-score)
        XGboost_index = XGboost_index[0:self.n_components]
        XGboost_index = np.array(XGboost_index).reshape(self.n_components,1)     #转换为2维数组
        score = sorted(score,reverse=True)     #降序排列
        score = score[0:self.n_components]     #提取前n_components个系数
        score = np.array(score).reshape(self.n_components,1)     #转换为2维数组
        score_index = np.hstack((score,XGboost_index))

        return score_index
