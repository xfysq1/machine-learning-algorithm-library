# -*- coding: utf-8 -*-

import numpy as np

from keras import backend as K
from keras.layers import Input, Dense
from keras.models import Model
from keras.callbacks import EarlyStopping

class SparseAutoencoder:

    def __init__(self, x, hidden_dims):
        
        self.x = x
        self.hidden_dims = np.array(hidden_dims)
        
    def construct_model(self, p=0.01, beta=1, encode_activation='sigmoid', decode_activation='sigmoid', use_linear=True):
        
        def sparse_constraint(activ_matrix):

            p_hat = K.mean(activ_matrix)
            KLD = p*(K.log(p/p_hat)) + (1-p)*(K.log(1-p/1-p_hat))
            
            return -beta * K.sum(KLD) # sum over the layer units
    
        input_layer = Input(shape=(self.x.shape[1], ))
        
        # AE
        if self.hidden_dims.shape[0] == 1:
            
            latent_layer = Dense(self.hidden_dims[0], activation = encode_activation, activity_regularizer=sparse_constraint)(input_layer)
            
            if use_linear == True:
                output_layer = Dense(self.x.shape[1], activation = 'linear')(latent_layer)
            else:
                output_layer = Dense(self.x.shape[1], activation = decode_activation)(latent_layer)
            
        # DAE
        else:
            
            encode_layer = Dense(self.hidden_dims[0], activation = encode_activation, activity_regularizer=sparse_constraint)(input_layer)
            for i in range(self.hidden_dims.shape[0]//2 - 1):
                encode_layer = Dense(self.hidden_dims[i + 1], activation = encode_activation, activity_regularizer=sparse_constraint)(encode_layer)
            
            latent_layer = Dense(self.hidden_dims[self.hidden_dims.shape[0]//2], activation = encode_activation, activity_regularizer=sparse_constraint)(encode_layer)
            
            decode_layer = Dense(self.hidden_dims[self.hidden_dims.shape[0]//2 + 1], activation = decode_activation, activity_regularizer=sparse_constraint)(latent_layer)
            for i in range(self.hidden_dims.shape[0]//2 - 1):
                decode_layer = Dense(self.hidden_dims[self.hidden_dims.shape[0]//2 + 2 + i], activation = decode_activation, activity_regularizer=sparse_constraint)(decode_layer)
            
            if use_linear == True:
                output_layer = Dense(self.x.shape[1], activation = 'linear')(decode_layer)
            else:
                output_layer = Dense(self.x.shape[1], activation = decode_activation)(decode_layer)
           
        self.SparseAutoencoder = Model(input=input_layer, output=output_layer)
        self.SparseEncoder = Model(input=input_layer, output=latent_layer)
        
    def train_model(self, epochs=1000, batch_size=100, optimizer='Adam', loss='mean_squared_error', use_Earlystopping=True):
        
        self.SparseAutoencoder.compile(optimizer=optimizer, loss=loss)
        
        if use_Earlystopping == True:
            self.history = self.SparseAutoencoder.fit(self.x, self.x, epochs = epochs, batch_size = batch_size, shuffle = True,
                                    validation_split = 0.10, callbacks = [EarlyStopping(monitor='val_loss', patience = 10)])
        else:
            self.history = self.SparseAutoencoder.fit(self.x, self.x, epochs = epochs, batch_size = batch_size, shuffle = True)
        
    def get_features(self, x_test):
        
        return self.SparseEncoder.predict(x_test)
        
    def get_reconstructions(self, x_test):
        
        return self.SparseAutoencoder.predict(x_test)
        
    def save_model(self,  SparseEncoder_name):

        #self.SparseEncoder.save(SparseEncoder_name + '.h5')
        self.SparseEncoder.save_weights(SparseEncoder_name + '.h5')

    def load_model(self, SparseEncoder_name):
        
        self.SparseEncoder.load_weights(SparseEncoder_name)

