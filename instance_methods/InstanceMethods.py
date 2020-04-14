#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 12:31:20 2020

@author: danielyaeger
"""
from sklearn.covariance import MinCovDet, LedoitWolf
from sklearn.neighbors import NearestNeighbors
from instance_methods.instance_method_helpers import mahalanobis
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
        
        n_jobs: number of processors to use. Select 1 for single processor.

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
        assert n_jobs >= 1, "n_jobs must be 0ne or greater!"
        self.k = k
        self.alpha = alpha
        self.n_jobs = n_jobs
        self.nn = NearestNeighbors(metric = 'euclidean', n_jobs = self.n_jobs)
    
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
        self.nn.fit(self.train_X)
       
        # Get distances. Set k += 1 because in training at least one distance will be zero
        distances = self.nn.kneighbors(self.train_X, n_neighbors = self.k + 1,
                                    return_distance = True)[0]
        # Calculate sum of square distances
        sum_of_square_distances = (distances*distances).sum(axis=1)
        # set threshold value based on prediction
        self.threshold = np.quantile(sum_of_square_distances, 1-self.alpha)
    
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
        # Get distances
        distances = self.nn.kneighbors(X, n_neighbors = self.k,
                                    return_distance = True)[0]
        # Calculate sum of square distances
        return (distances*distances).sum(axis=1)
    
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
        sum_of_square_distances = self.decision_function(X)
        pred_y[sum_of_square_distances > self.threshold] = outlier_label
        return pred_y
    
    def kneighbors(self, X: np.ndarray, n_neighbors: int = 5) -> np.ndarray:
        """Returns the sum-of-square distances to the nearest n_neighbors for
        every row in X.
        """
        distances = self.nn.kneighbors(X, n_neighbors = n_neighbors,
                                    return_distance = True)[0]
        return (distances*distances).sum(axis=1)
    
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
    def __init__(self, K: int = 500, k: int = 3, alpha: float = 0.01, 
                 precision_method = 'LedoitWolf', n_jobs: int = 4):
        assert 0 < alpha < 1, "alpha must be between 0 and 1!"
        assert 1 < k < K, "K and k must be greater than 1 and K must be greater than k!"
        self.K = K
        self.k = k
        self.alpha = alpha
        self.n_jobs = n_jobs
        
        if precision_method == 'LedoitWolf':
            self.precision_method = LedoitWolf
        elif precision_method == 'MinCovDet':
            self.precision_method = MinCovDet
        
        self.nn = NearestNeighbors(metric = 'euclidean', n_jobs = self.n_jobs)
    
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
        """Wrapper function for finding nearest K samples in Euclidean space,
        calculating precision matrices, and then finding nearesrt k samples
        using mahalanobis_distances. Returns array of summed mahalanobis
        distances.
        """
        # fit on training data if train_mode = True
        if train_mode:
            self.nn.fit(X)
            K = self.K + 1
            k = self.k + 1
        
        # Find nearest samples
        nearest_neighbors = self.nn.kneighbors(X, n_neighbors = K,
                                    return_distance = False)
        if train_mode:
            # The first entry in each row is the sample itself
            nearest_neighbors = nearest_neighbors[:,1:]
        
        # Compute precision matrices
        precision_matrices = np.zeros((nearest_neighbors.shape[0], X.shape[1], X.shape[1]))
        
        for i in range(nearest_neighbors.shape[0]):
            precision_matrices[i,:,:] = self.precision_method().fit(
                    X[nearest_neighbors[i,:],:]).get_precision()
        
        # Find sum of distances to nearest neighbor in mahalanobis distances
        mahalanobis_distances = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            X_mahalanobis = mahalanobis(sample_to_score = X[i,:],
                                        reference_array = self.train_X,
                                        precision_matrix = precision_matrices[i,:,:])
            knn = NearestNeighbors(metric = 'euclidean', n_jobs = self.n_jobs)
            knn.fit(X_mahalanobis)
            distances = knn.kneighbors(X_mahalanobis, n_neighbors = k,
                                    return_distance = True)[0]
            mahalanobis_distances[i] = np.sqrt(distances.sum(axis=1))
        return mahalanobis_distances
        
    
    
    