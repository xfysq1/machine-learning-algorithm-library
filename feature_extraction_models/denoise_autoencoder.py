# -*- coding: utf-8 -*-

import numpy as np

from keras.layers import Input, Dense
from keras.models import Model
from keras.models import load_model
from keras.callbacks import EarlyStopping

class DenoiseAutoencoder:

    def __init__(self, x, hidden_dims, corrupt='Gaussian', corrupt_rate=0.5):
        
        self.x = x
        if corrupt == 'Gaussian':
            self.x_corrupted = x + corrupt_rate * np.random.normal(loc=0.0, scale=1.0, size=x.shape)
        if corrupt == 'binary':
            self.x_corrupted = x * np.random.binomial(n=1, p=corrupt_rate, size=x.shape)
        self.hidden_dims = np.array(hidden_dims)
        
    def construct_model(self, encode_activation='sigmoid', decode_activation='sigmoid', use_linear=True):
    
        input_layer = Input(shape=(self.x.shape[1], ))
        
        # AE
        if self.hidden_dims.shape[0] == 1:
            
            latent_layer = Dense(self.hidden_dims[0], activation = encode_activation)(input_layer)
            
            if use_linear == True:
                output_layer = Dense(self.x.shape[1], activation = 'linear')(latent_layer)
            else:
                output_layer = Dense(self.x.shape[1], activation = decode_activation)(latent_layer)
            
        # DAE
        else:
            
            encode_layer = Dense(self.hidden_dims[0], activation = encode_activation)(input_layer)
            for i in range(self.hidden_dims.shape[0]//2 - 1):
                encode_layer = Dense(self.hidden_dims[i + 1], activation = encode_activation)(encode_layer)
            
            latent_layer = Dense(self.hidden_dims[self.hidden_dims.shape[0]//2], activation = encode_activation)(encode_layer)
            
            decode_layer = Dense(self.hidden_dims[self.hidden_dims.shape[0]//2 + 1], activation = decode_activation)(latent_layer)
            for i in range(self.hidden_dims.shape[0]//2 - 1):
                decode_layer = Dense(self.hidden_dims[self.hidden_dims.shape[0]//2 + 2 + i], activation = decode_activation)(decode_layer)
            
            if use_linear == True:
                output_layer = Dense(self.x.shape[1], activation = 'linear')(decode_layer)
            else:
                output_layer = Dense(self.x.shape[1], activation = decode_activation)(decode_layer)
           
        self.DenoiseAutoencoder = Model(input=input_layer, output=output_layer)
        self.DenoiseEncoder = Model(input=input_layer, output=latent_layer)
        
    def train_model(self, epochs=1000, batch_size=100, optimizer='Adam', loss='mean_squared_error', use_Earlystopping=True):
        
        self.DenoiseAutoencoder.compile(optimizer=optimizer, loss=loss)
        
        if use_Earlystopping == True:
            self.history = self.DenoiseAutoencoder.fit(self.x_corrupted, self.x, epochs = epochs, batch_size = batch_size, shuffle = True,
                                    validation_split = 0.10, callbacks = [EarlyStopping(monitor='val_loss', patience = 30)])
        else:
            self.history = self.DenoiseAutoencoder.fit(self.x, self.x, epochs = epochs, batch_size = batch_size, shuffle = True)
        
    def get_features(self, x_test):
        
        return self.DenoiseEncoder.predict(x_test)
        
    def get_reconstructions(self, x_test):
        
        return self.DenoiseAutoencoder.predict(x_test)
        
    def save_model(self, DenoiseEncoder_name=None):

        #self.DenoiseEncoder.save(DenoiseEncoder_name + '.h5')
        self.DenoiseEncoder.save_weights(DenoiseEncoder_name + '.h5')
        
    def load_model(self, DenoiseEncoder_name=None):

        #self.DenoiseEncoder = load_model(DenoiseEncoder_name)
        self.DenoiseEncoder.load_weights(DenoiseEncoder_name)