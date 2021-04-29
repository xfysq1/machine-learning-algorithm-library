import math
import numpy as np
from sklearn import preprocessing
from sklearn import metrics
import pandas as pd
from keras import backend as K
from keras.layers import Input, Dense
from keras.models import Model
from keras.callbacks import EarlyStopping
from keras import regularizers

class fs_SNN:

    def __init__(self, z, n_components, hidden_dims, epochs, batch_size):

        self.z = z     #数据矩阵
        self.n_components = n_components
        self.hidden_dims = hidden_dims     #隐含层神经元数量
        self.epochs = epochs
        self.batch_size = batch_size


    def construct_SNN_model(self, p=0.01, beta=1, encode_activation='sigmoid', decode_activation='sigmoid',
                            use_linear=True, optimizer='Adam', loss='mean_squared_error',
                            use_Earlystopping=True):

        print(1)
        dataSet = self.z
        dataSet = np.array(dataSet)
        train_data = dataSet[:,0:-1]
        train_label = dataSet[:,-1]

        def sparse_constraint(activ_matrix):     #定义稀疏约束函数

            p_hat = K.mean(activ_matrix)
            KLD = p * (K.log(p / p_hat)) + (1 - p) * (K.log(1 - p / 1 - p_hat))

            return -beta * K.sum(KLD)  # sum over the layer units

        input_layer = Input(shape=(train_data.shape[1],))     #定义输入层神经元个数，为输入数据的列数

        latent_layer = Dense(self.hidden_dims, activation=encode_activation,
                             activity_regularizer=regularizers.l1(10e-5))(input_layer)     #连接隐藏层与输入层

        if use_linear:     #连接输出层和隐藏层
            output_layer = Dense(1, activation='linear')(latent_layer)
        else:
            output_layer = Dense(1, activation=decode_activation)(latent_layer)

        self.SparseAutoencoder = Model(input=input_layer, output=output_layer)

        #训练模型
        self.SparseAutoencoder.compile(optimizer=optimizer, loss=loss)
        self.SparseAutoencoder.fit(train_data, train_label, epochs=self.epochs, batch_size=self.batch_size,
                                                  shuffle=True)

        if use_Earlystopping == True:
            self.history = self.SparseAutoencoder.fit(train_data, train_label, epochs = self.epochs, batch_size = self.batch_size, shuffle = True,
                                    validation_split = 0.10, callbacks = [EarlyStopping(monitor='val_loss', patience = 10)])
        else:
            self.history = self.SparseAutoencoder.fit(train_data, train_label, epochs = self.epochs, batch_size = self.batch_size, shuffle = True)

        w1,b1,w2,b2 = self.SparseAutoencoder.get_weights()     #获取权值
        w1 = np.array(w1)
        w2 = np.array(w2)
        score = np.dot(w1,w2)
        score = np.ravel(score)
        score = np.fabs(score)
        SNN_index = np.argsort(-score)
        SNN_index = SNN_index[0:self.n_components]
        SNN_index = np.array(SNN_index).reshape(self.n_components,1)
        score = sorted(score,reverse=True)
        score = score[0:self.n_components]
        score = np.array(score).reshape(self.n_components,1)

        return score,SNN_index


    def extract_SNN_feature(self):

        w1,b1,w2,b2 = self.SparseAutoencoder.get_weights()
        w1 = np.array(w1)
        w2 = np.array(w2)
        score = np.dot(w1,w2)
        score = np.ravel(score)
        score = np.fabs(score)
        SNN_index = np.argsort(-score)
        SNN_index = SNN_index[0:self.n_components]
        SNN_index = np.array(SNN_index).reshape(self.n_components,1)
        score = sorted(score,reverse=True)
        score = score[0:self.n_components]
        score = np.array(score).reshape(self.n_components,1)
        score_index = np.hstack((score,SNN_index))

        return score_index








