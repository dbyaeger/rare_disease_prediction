#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 15:39:06 2020

@author: yaeger
"""
from sklearn.base import BaseEstimator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import numpy as np

class AutoEncoder(BaseEstimator):
    """Constructs a one-hidden layer autoencoder for anomaly detection
    """
    def __init__(self, num_hidden: int = 1024, lambda_hidden: float = 1e-6, 
                 dropout_p: float = 0.5, use_dropout: bool = False, 
                 train_epochs: int = 100, hidden_activation_function: str = 'relu',
                 batch_size: int = 128, learning_rate: float = 0.001,
                 batch_norm: bool = True, alpha: float = 0.999):
        self.num_hidden = num_hidden
        self.lambda_hidden = lambda_hidden
        self.dropout_p = dropout_p
        self.use_dropout = use_dropout
        self.train_epochs = train_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.hidden_activation_function = hidden_activation_function
        self.batch_norm = batch_norm
        self.alpha = alpha
        
    def fit(self,X,y=None):
        """Creates and fits a model"""
        
        # Make model
        input_layer = Input(shape = (X.shape[-1], ))
        encoder = Dense(self.num_hidden, 
                        activation = self.hidden_activation_function, 
                        kernel_regularizer = l2(self.lambda_hidden))(input_layer)
        if self.batch_norm:
            encoder = BatchNormalization()(encoder)
        if self.use_dropout:
            encoder = Dropout(self.dropout_p)(encoder)
        decoder = Dense(X.shape[-1], activation = 'sigmoid')(encoder)
        self.autoencoder_ = Model(inputs=input_layer, outputs = decoder)
        
        # Compile model
        optim = Adam(self.learning_rate)
        self.autoencoder_.compile(optimizer=optim,
                            loss = "mean_squared_error")
        
        # Train model
        self.autoencoder_.fit(X,X,
                  batch_size = self.batch_size,
                  epochs = self.train_epochs,
                  verbose = 2)
        
        # Build the error distribution
        predicted = self.autoencoder_.predict(X)
        
        self.error_distribution_ = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            self.error_distribution_[i] = ((predicted[i,:] - X[i,:]).T)@(predicted[i,:] - X[i,:])
        
        # Calculate a threshold
        self.threshold_ = np.quantile(self.error_distribution_,self.alpha)
        
    def predict(self, X, outlier_label = 1, majority_label = -1):
        """Predicts the label for each input X"""
        pred_y = np.ones(X.shape[0])*majority_label
        
        predicted = self.autoencoder_.predict(X)
        
        mse = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            mse[i] = ((predicted[i,:] - X[i,:]).T)@(predicted[i,:] - X[i,:])
        
        pred_y[mse > self.threshold_] = outlier_label
        return pred_y
    
    def decision_function(self, X: np.ndarray):
        """Returns reconstruction error for each observation in X
        """
        predicted = self.autoencoder_.predict(X)
        
        mse = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            mse[i] = ((predicted[i,:] - X[i,:]).T)@(predicted[i,:] - X[i,:])

        return mse
        
        
        
                        
                        
    