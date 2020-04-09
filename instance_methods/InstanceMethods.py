#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 12:31:20 2020

@author: danielyaeger
"""
from sklearn.covariance import MinCovDet, LedoitWolf

from instance_methods.instance_method_helpers import (kNN_distances, 
                                                      KNN_samples,
                                                      knn_mahalanobis)
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
    def __init__(self, k: int = 3, alpha: float = 0.01):
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
        self.train_X = X[y == majority_label,:].copy()
        self.sum_of_square_distances = kNN_distances(array_to_score = self.train_X,
                                      reference_array = self.train_X,
                                      k = self.k, train_mode = True)
        
        # set threshold value based on prediction
        self.threshold = np.quantile(self.sum_of_square_distances, 1-self.alpha)
    
    def kneighbors(self, X: np.ndarray, n_neighbors: int = 5) -> np.ndarray:
        """Returns the sum-of-square distances to the nearest n_neighbors for
        every row in X.
        """
        
        if X.shape[0] == 1:
            X = np.array(X)
            
        return kNN_distances(array_to_score = X,
                             reference_array = self.train_X,
                             k = n_neighbors, train_mode = False)
        

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
                                      k = self.k, train_mode = False)
        pred_y[sum_of_square_distances > self.threshold] = outlier_label
        return pred_y

    def decision_function(self, X: np.ndarray):
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
                             k = self.k, train_mode = False)

class MahalanobisDistanceKNN():
    """Implementation of the Adaptive Mahalanobis Distance KNN described in
    "Adaptive Mahalanobis Distance and k-Nearest Neighbor Rule for Fault
    Detection in Semiconductor Manufacturing" by Verdier and Ferreira.
    
    NOTE: For binary classification only.

    PARAMETERS:
        K: number of nearest neighbors based on Euclidean space used to 
        estimate the covariance matrix.
        
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
    def __init__(self, K: int = 500, k: int = 3, alpha: float = 0.01, precision_method = 'LedoitWolf'):
        assert 0 < alpha < 1, "alpha must be between 0 and 1!"
        assert 1 < k < K, "K and k must be greater than 1 and K must be greater than k!"
        self.K = K
        self.k = k
        self.alpha = alpha
        
        if precision_method == 'LedoitWolf':
            self.precision_method = LedoitWolf
        elif precision_method == 'MinCovDet':
            self.precision_method = MinCovDet
    
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
        self.train_X = X[y == majority_label,:].copy()
        
     
        # Find sum of distances to nearest neighbor in mahalanobis distances
        sum_of_mahalanobis_distances = self.get_mahalanobis_distances(X = self.train_X,
                                                                 train_mode = True,
                                                                 k = self.k)

        # set threshold value based on prediction
        self.threshold = np.quantile(sum_of_mahalanobis_distances, 1-self.alpha)
    
    def kneighbors(self, X: np.ndarray, n_neighbors: int = 5) -> np.ndarray:
        """Returns the sum-of-square distances to the nearest n_neighbors for
        every row in X.
        """
        if X.shape[0] == 1:
            X = np.array(X)
            
        # Find sum of distances to nearest neighbor in mahalanobis distances
        return self.get_mahalanobis_distances(X = X, train_mode = False, k = n_neighbors)
        

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
        sum_of_mahalanobis_distances = self.get_mahalanobis_distances(X = X,
                                                                 train_mode = False,
                                                                 k = self.k)
        pred_y[sum_of_mahalanobis_distances > self.threshold] = outlier_label
        return pred_y

    def decision_function(self, X: np.ndarray):
        """Returns sum-of-square distances to k nearest neighbors based on
        input array.

        INPUTS:
            X: array of training data, with features as columns and observations
            in rows.
        RETURNS:
            distances: array of sum-of-square distances to k nearest neighbor
            for each observation in X.
        """
        return self.get_mahalanobis_distances(X = X, train_mode = False, k = self.k)
    
    def get_mahalanobis_distances(self, X: np.array, train_mode: bool, k: int):
        """Wrapper function for finding neasrest K samples in Euclidean space,
        calculating precision matrices, and then finding nearesrt k samples
        using mahalanobis_distances. Returns array of summed mahalanobis
        distances.
        """
        #Find nearest samples in Euclidean space
        nearest_obs_matrix = KNN_samples(array_to_score = X, 
                                           reference_array = self.train_X, 
                                           K = self.K,
                                           train_mode = train_mode)
        
        # Compute precision matrices
        precision_matrices = np.zeros((nearest_obs_matrix.shape[0], self.train_X.shape[1], self.train_X.shape[1]))
        
        for i in range(nearest_obs_matrix.shape[0]):
            precision_matrices[i,:,:] = self.precision_method().fit(nearest_obs_matrix[i,:,:]).get_precision()
        
        # Find sum of distances to nearest neighbor in mahalanobis distances
        return knn_mahalanobis(array_to_score = X, reference_array = self.train_X,
                               precision_matrices = precision_matrices, 
                               k = k, train_mode = train_mode) 
