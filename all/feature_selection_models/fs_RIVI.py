import math
import numpy as np
from sklearn import preprocessing
from sklearn import metrics
import pandas as pd


class fs_RIVI:

    def __init__(self, z, n_components, preprocess=True):

        self.z = z
        self.preprocess = preprocess
        self.n_components = n_components

        if self.preprocess:
            self.Xscaler = preprocessing.StandardScaler().fit(self.z)
            self.z = self.Xscaler.transform(self.z)

    def construct_RIVI_model(self):

        dataSet = np.array(self.z)
        dataSet = np.array(dataSet)
        X0 = dataSet[:, 0:-1]
        Y0 = dataSet[:, -1]

        n, m = np.shape(X0)
        errorfull = np.zeros(n)

        for i in range(n):
            Xtrain = X0
            Ytrain = Y0
            Xtrain = np.delete(Xtrain, i, axis=0)
            Ytrain = np.delete(Ytrain, i, axis=0)
            Xtest = X0[i, :]
            Ytest = Y0[i]
            K = np.dot(Xtrain, Xtrain.T)
            delta = np.var(Ytrain)
            I = np.eye(n - 1, n - 1)
            Kq = np.dot(Xtest, Xtrain.T)
            t1 = np.linalg.inv(K + np.dot(delta, I))
            tt1 = np.dot(t1, Ytrain)
            Yp = np.dot(Kq, tt1)
            errorfull[i] = Ytest - Yp

        errorm = np.zeros([n, m])
        for j in range(m):
            X1 = X0
            Y1 = Y0
            X1 = np.delete(X1, j, axis=1)
            for i in range(n):
                Xtrain = X1
                Ytrain = Y1
                Xtrain = np.delete(Xtrain, i, axis=0)
                Ytrain = np.delete(Ytrain, i, axis=0)
                Xtest = X1[i, :]
                Ytest = Y1[i]
                K = np.dot(Xtrain, Xtrain.T)
                delta = np.var(Ytrain)
                I = np.eye(n - 1, n - 1)
                Kq = np.dot(Xtest, Xtrain.T)
                t2 = np.linalg.inv(K + np.dot(delta, I))
                tt2 = np.dot(t2, Ytrain)
                Yp = np.dot(Kq, tt2)
                errorm[i, j] = Ytest - Yp

        IH = np.eye(n)
        Ky = IH
        Lf = IH
        Lm = np.zeros([n, n, m])
        for i in range(n):
            for j in range(i, n):
                t3 = Y0[i] - Y0[j]
                tt3 = -t3**2/2
                Ky[i, j] = np.exp(tt3)
                Ky[j, i] = Ky[i, j]

        for i in range(n):
            for j in range(i, n):
                t4 = np.dot(errorfull[i], errorfull[j])
                Lf[i, j] = t4 ** 2
                Lf[j, i] = Lf[i, j]

        for k in range(m):
            for i in range(n):
                for j in range(i, n):
                    t5 = np.dot(errorm[i, k], errorm[j, k])
                    Lm[i, j, k] = t5 ** 2
                    Lm[j, i, k] = Lm[i, j, k]

        H = IH - 1 / n
        t6 = np.dot(Ky, H)
        tt6 = np.dot(Lf, H)
        ttt6 = np.dot(t6, tt6)
        HSICfull = np.trace(ttt6)
        HSICm = np.zeros(m)
        score = np.zeros(m)
        for k in range(m):
            t7 = np.dot(Ky, H)
            tt7 = np.dot(Lm[:, :, k], H)
            ttt7 = np.dot(t7, tt7)
            HSICm[k] = np.trace(ttt7)
            score[k] = HSICm[k] / HSICfull

        score = np.fabs(score)     #对计算得到的系数取绝对值
        RIVI_index = np.argsort(-score)     #获取降序排列的索引
        RIVI_index = RIVI_index[0:self.n_components]     #提取前n_components个索引
        score = sorted(score, reverse=True)  # 降序排列
        score= score[0:self.n_components]  # 提取前n_components个系数

        return score,RIVI_index

    def extract_RIVI_feature(self, x_test):
        p2 = []
        x_test = self.Xscaler.transform(x_test)
        dataSet = x_test
        dataSet = np.array(dataSet)
        X0 = dataSet[:, 0:-1]
        Y0 = dataSet[:, -1]

        n, m = np.shape(X0)
        errorfull = np.zeros(n)

        for i in range(n):
            Xtrain = X0
            Ytrain = Y0
            Xtrain = np.delete(Xtrain, i, axis=0)
            Ytrain = np.delete(Ytrain, i, axis=0)
            Xtest = X0[i, :]
            Ytest = Y0[i]
            K = np.dot(Xtrain, Xtrain.T)
            delta = np.var(Ytrain)
            I = np.eye(n - 1, n - 1)
            Kq = np.dot(Xtest, Xtrain.T)
            t1 = np.linalg.inv(K + np.dot(delta, I))
            tt1 = np.dot(t1, Ytrain)
            Yp = np.dot(Kq, tt1)
            errorfull[i] = Ytest - Yp

        errorm = np.zeros([n, m])
        for j in range(m):
            X1 = X0
            Y1 = Y0
            X1 = np.delete(X1, j, axis=1)
            for i in range(n):
                Xtrain = X1
                Ytrain = Y1
                Xtrain = np.delete(Xtrain, i, axis=0)
                Ytrain = np.delete(Ytrain, i, axis=0)
                Xtest = X1[i, :]
                Ytest = Y1[i]
                K = np.dot(Xtrain, Xtrain.T)
                delta = np.var(Ytrain)
                I = np.eye(n - 1, n - 1)
                Kq = np.dot(Xtest, Xtrain.T)
                t2 = np.linalg.inv(K + np.dot(delta, I))
                tt2 = np.dot(t2, Ytrain)
                Yp = np.dot(Kq, tt2)
                errorm[i, j] = Ytest - Yp

        IH = np.eye(n)
        Ky = IH
        Lf = IH
        Lm = np.zeros([n, n, m])
        for i in range(n):
            for j in range(i, n):
                t3 = Y0[i] - Y0[j]
                tt3 = -t3**2/2
                Ky[i, j] = np.exp(tt3)
                Ky[j, i] = Ky[i, j]

        for i in range(n):
            for j in range(i, n):
                t4 = np.dot(errorfull[i], errorfull[j])
                Lf[i, j] = t4 ** 2
                Lf[j, i] = Lf[i, j]

        for k in range(m):
            for i in range(n):
                for j in range(i, n):
                    t5 = np.dot(errorm[i, k], errorm[j, k])
                    Lm[i, j, k] = t5 ** 2
                    Lm[j, i, k] = Lm[i, j, k]

        H = IH - 1 / n
        t6 = np.dot(Ky, H)
        tt6 = np.dot(Lf, H)
        ttt6 = np.dot(t6, tt6)
        HSICfull = np.trace(ttt6)
        HSICm = np.zeros(m)
        score = np.zeros(m)
        for k in range(m):
            t7 = np.dot(Ky, H)
            tt7 = np.dot(Lm[:, :, k], H)
            ttt7 = np.dot(t7, tt7)
            HSICm[k] = np.trace(ttt7)
            score[k] = HSICm[k] / HSICfull

        score = np.fabs(score)     #对计算得到的系数取绝对值
        RIVI_index = np.argsort(-score)     #获取降序排列的索引
        RIVI_index = RIVI_index[0:self.n_components]     #提取前n_components个索引
        RIVI_index = np.array(RIVI_index).reshape(self.n_components,1)     #转换为2维数组
        score = sorted(score, reverse=True)  # 降序排列
        score= score[0:self.n_components]  # 提取前n_components个系数
        score = np.array(score).reshape(self.n_components,1)     #转换为2维数组
        score_index = np.hstack((score,RIVI_index))     #水平合并


        return score_index



