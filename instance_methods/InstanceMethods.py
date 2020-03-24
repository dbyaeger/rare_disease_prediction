#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 12:31:20 2020

@author: danielyaeger
"""
from instance_methods.instance_method_helpers import kNN_distances
import numpy as np

class FaultDetectionKNN():
    """Implementation of the classifier in "Fault Detection Using the k-Nearest 
    Neighbor Rule for Semiconductor Manufacturing Processes" by He and Wang.
    
    NOTE: For binary classification only.
    
    PARAMETERS:
        k: number of k-nearest neighbors
        
        alpha: specifier for a threshold, defined as 1-alpha, which sets
            the quantile over the training data above which a sample must
            fall to be classified as abnormal. NOTE: only negative training
            data is used in building this distribution.
    
    METHODS:
        fit: takes in training data array X and labels y and generates 
        distribution of squared distances to k-nearest neighbors for each
        point. Only uses data from the negative class to build the distribution.
    
        predict: takes in an array of data and predicts the class for each 
        sample.
        
        decision_function: takes in an array of data and returns the 
        sum-of-square distances of each point's k nearest neighbors

    """
    def __init__(self, k: int = 3, alpha: float = 0.01, n_jobs: int = 4):
        assert 0 < alpha < 1, "alpha must be between 0 and 1!"
        self.k = k
        self.alpha = alpha
        
    def fit(self,X: np.ndarray,y: np.ndarray, majority_label: int = -1):
        """ Generates distribution of sum-of-square distance to k nearest
        neighbors.
        
        INPUTS:
            X: array of training data, with features as columns and observations
            in rows
            y: training data labels
            majority_label: label of data on which to train, by default set to
            -1. Data without this label will be discarded.
        RETURNS:
            None
        """
        self.train_X = X[y == majority_label,:]
        sum_of_square_distances = kNN_distances(array_to_score = self.train_X,
                                      reference_array = self.train_X,
                                      k = self.k)
        
        # set threshold value based on prediction
        self.threshold = np.quantile(sum_of_square_distances, 1-self.alpha)

    def predict(self, X, outlier_label = 1, majority_label = -1):
        """Returns predicted class based on input array.
        
        INPUTS: 
            X: array of training data, with features as columns and observations
            in rows.
            outlier_label: label for abnormal samples. Default of 1
            majority_label: label for normal samples. Default of -1
        RETURNS:
            pred_y: array of predicted classes
        """
        pred_y = np.ones(X.shape[0])*majority_label
        sum_of_square_distances = kNN_distances(array_to_score = X,
                                      reference_array = self.train_X,
                                      k = self.k)
        pred_y[sum_of_square_distances > self.threshold] = outlier_label
        return pred_y
    
    def decision_function(self, X):
        """Returns sum-of-square distances to k nearest neighbors based on 
        input array.
        
        INPUTS: 
            X: array of training data, with features as columns and observations
            in rows.
        RETURNS:
            distances: array of sum-of-square distances to k nearest neighbor
            for each observation in X.
        """
        return kNN_distances(array_to_score = X,
                             reference_array = self.train_X,
                             k = self.k)

        
            
            