# -*- coding: utf-8 -*-
import math
import numpy as np
from sklearn import preprocessing
import pandas as pd
from sklearn import metrics

class fs_MRMR:

    def __init__(self, z, n_components, preprocess=True):

        self.z = z
        self.preprocess = preprocess
        self.n_components = n_components

        if self.preprocess:
            self.Xscaler = preprocessing.StandardScaler().fit(self.z)
            self.z = self.Xscaler.transform(self.z)

    def getmultimi(self, da, dt):
        c = []
        da = np.array(da).reshape(self.n, 1)
        dt = np.array(dt).reshape(self.n, 1)
        for i in range(da.shape[1]):
            da = np.ravel(da)
            dt = np.ravel(dt)
            c.append(metrics.normalized_mutual_info_score(da, dt, average_method='max'))
        c = np.array(c)
        return c

    def construct_MRMR_model(self):

        p = []
        dataSet = self.z
        dataSet = np.array(dataSet)
        self.n,self.m = np.shape(dataSet)
        x = [0]*self.n  # 初始化特征x和类别y向量
        y = [0]*self.n
        d = dataSet[:, 0:-1]
        #计算初始互信息值
        for i in range(self.n):  # 得到类向量
            y[i] = dataSet[i][self.m-1]
        for j in range(self.m - 1):  # 获取每个特征的向量
            for k in range(self.n):
                x[k] = dataSet[k][j]
            p.append(metrics.normalized_mutual_info_score(x, y, average_method='max'))

        p = np.array(p)
        tmppp = sorted(p, reverse=True)     #对初始互信息值降序排列
        idxs = np.argsort(-p)     #获取降序排列的索引值
        fea_base = idxs[0:self.n_components]

        fea1 = []     #用于存储MRMR的索引
        tmp1 = []     #用于存储MRMR的系数
        fea1.append(idxs[0])
        tmp1.append(tmppp[0])

        KMAX = min(1000, self.m - 1)  #为防止特征系数过多，规定了最大特征个数

        idxleft = idxs[1:KMAX]
        idxleft = np.array(idxleft)

        t_mi = np.zeros([1, self.m-2])     #初始化
        mi_array = np.zeros([self.m-1, self.m-2])
        c_mi = np.zeros([1, self.m-2])
        y = np.array(y)

        for k in range(1,self.n_components):

           ncand = np.size(idxleft)
           curlastfea = np.size(fea1)-1
           for i in range(ncand):
              #计算按照初始互信息降序后的索引值对应的数据与标签列的互信息系数
              t_mi[0,i] = metrics.normalized_mutual_info_score(d[:,idxleft[i]], y, average_method='max')
              #计算初始互信息按降序顺序的各较大值索引对应的列与后序列的互信息
              mi_array[idxleft[i], curlastfea] = self.getmultimi(d[:, fea1[curlastfea]], d[:, idxleft[i]])
              c_mi[0,i] = np.mean(mi_array[idxleft[i], :])

           mv = t_mi[0:ncand] - c_mi[0:ncand]
           mv = np.ravel(mv)
           mv = mv[0:ncand]
           tmp1.append(max(mv))
           fea1.append(np.argmax(mv))

           tmpidx = fea1[k]
           fea1[k] = idxleft[tmpidx]
           idxleft = np.delete(idxleft, tmpidx)
        tmp1 = np.array(tmp1)
        tmp1 = np.fabs(tmp1)
        fea1 = np.array(fea1)
        score = tmp1
        MRMR_index = fea1
        return score,MRMR_index

    def extract_MRMR_feature(self, x_test):

        p = []
        x_test = self.Xscaler.transform(x_test)
        dataSet = x_test
        dataSet = np.array(dataSet)
        self.n,self.m = np.shape(dataSet)
        x = [0]*self.n  # 初始化特征x和类别y向量
        y = [0]*self.n
        d = dataSet[:, 0:-1]
        #计算初始互信息值
        for i in range(self.n):  # 得到类向量
            y[i] = dataSet[i][self.m-1]
        for j in range(self.m - 1):  # 获取每个特征的向量
            for k in range(self.n):
                x[k] = dataSet[k][j]
            p.append(metrics.normalized_mutual_info_score(x, y, average_method='max'))

        p = np.array(p)
        tmppp = sorted(p, reverse=True)     #对初始互信息值降序排列
        idxs = np.argsort(-p)     #获取降序排列的索引值
        fea_base = idxs[0:self.n_components]

        fea2 = []     #用于存储MRMR的索引
        tmp2 = []     #用于存储MRMR的系数
        fea2.append(idxs[0])
        tmp2.append(tmppp[0])

        KMAX = min(1000, self.m - 1)  #为防止特征系数过多，规定了最大特征个数

        idxleft = idxs[1:KMAX]
        idxleft = np.array(idxleft)

        t_mi = np.zeros([1, self.m-2])     #初始化
        mi_array = np.zeros([self.m-1, self.m-2])
        c_mi = np.zeros([1, self.m-2])
        y = np.array(y)

        for k in range(1,self.n_components):

           ncand = np.size(idxleft)
           curlastfea2 = np.size(fea2)-1
           for i in range(ncand):
              #计算按照初始互信息降序后的索引值对应的数据与标签列的互信息系数
              t_mi[0,i] = metrics.normalized_mutual_info_score(d[:,idxleft[i]], y, average_method='max')
              #计算初始互信息按降序顺序的各较大值索引对应的列与后序列的互信息
              mi_array[idxleft[i], curlastfea2] = self.getmultimi(d[:, fea2[curlastfea2]], d[:, idxleft[i]])
              c_mi[0,i] = np.mean(mi_array[idxleft[i], :])

           mv = t_mi[0:ncand] - c_mi[0:ncand]
           mv = np.ravel(mv)
           mv = mv[0:ncand]
           tmp2.append(max(mv))
           fea2.append(np.argmax(mv))

           tmpidx = fea2[k]
           fea2[k] = idxleft[tmpidx]
           idxleft = np.delete(idxleft, tmpidx)
        tmp2 = np.array(tmp2).reshape(self.n_components,1)
        tmp2 = np.fabs(tmp2)
        fea2 = np.array(fea2).reshape(self.n_components,1)
        score = tmp2
        MRMR_index = fea2
        score_index = np.hstack((score,MRMR_index))

        return score_index


