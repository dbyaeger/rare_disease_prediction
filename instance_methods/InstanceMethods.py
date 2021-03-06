#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 12:31:20 2020

@author: danielyaeger
"""
from sklearn.covariance import MinCovDet, LedoitWolf, EmpiricalCovariance
from sklearn.neighbors import NearestNeighbors
from instance_methods.instance_method_helpers import mahalanobis
import numpy as np
from sklearn.base import BaseEstimator

class FaultDetectionKNN(BaseEstimator):
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
        
        # Make sure k is an integer
        if not isinstance(k,int): k = int(k)
        
        self.k = k
        self.alpha = alpha
        self.n_jobs = n_jobs
        self.nn_ = NearestNeighbors(metric = 'euclidean', n_jobs = self.n_jobs)
    
    def fit(self,X: np.ndarray,y: np.ndarray = None, majority_label: int = -1):
        """ Generates distribution of sum-of-square distance to k nearest
        neighbors.

        INPUTS:
            X: array of training data, with features as columns and observations
            in rows
            y: training data labels. Optional - if set to None it is assumed all
            data are negative instances.
            majority_label: label of data on which to train, by default set to
            -1. Data without this label will be discarded.
        RETURNS:
            None
        """
        if y is not None:
            train_X = X[y == majority_label,:].copy()
        else:
            train_X = X.copy()
            
        self.nn_.fit(train_X)
       
        # Get distances. Set k += 1 because in training at least one distance will be zero
        distances = self.nn_.kneighbors(train_X, n_neighbors = self.k + 1,
                                    return_distance = True)[0]
        # Calculate sum of square distances
        sum_of_square_distances = (distances*distances).sum(axis=1)
        # set threshold value based on prediction
        self.threshold_ = np.quantile(sum_of_square_distances, 1-self.alpha)
    
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
        distances = self.nn_.kneighbors(X, n_neighbors = self.k,
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
        pred_y[sum_of_square_distances > self.threshold_] = outlier_label
        return pred_y
    
    def kneighbors(self, X: np.ndarray, n_neighbors: int = 5) -> np.ndarray:
        """Returns the sum-of-square distances to the nearest n_neighbors for
        every row in X.
        """
        if not isinstance(n_neighbors, int): 
            n_neighbors = int(n_neighbors)
        
        distances = self.nn_.kneighbors(X, n_neighbors = n_neighbors,
                                    return_distance = True)[0]
        
        return (distances*distances).sum(axis=1)
    
class MahalanobisDistanceKNN(BaseEstimator):
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
    def __init__(self, K: int = 500, k: int = 3, alpha: float = 0.01, n_jobs: int = 4):
        assert 0 < alpha < 1, "alpha must be between 0 and 1!"
        assert 1 < k < K, "K and k must be greater than 1 and K must be greater than k!"
                
        # sci-kit learn constructor cannot modify values set by user        
        self.K = K
        self.k = k
        self.alpha = alpha
        self.n_jobs = n_jobs
        
        # "... constructor in an estimator should only set attributes to the values 
        # the user passes as arguments. All All computation should occur in fit,
        # and if fit needs to store the result of a computation, 
        # it should do so in an attribute with a trailing underscore (_)."
        # https://stackoverflow.com/questions/24510510/python-scikit-learn-cannot-clone-object-as-the-constructor-does-not-seem-to
        self.precision_method_ = EmpiricalCovariance
        
        self.nn_ = NearestNeighbors(metric = 'euclidean', n_jobs = self.n_jobs)
    
    def fit(self,X: np.ndarray,y: np.ndarray= None, majority_label: int = -1):
        """ Generates distribution of sum-of-square distance to k nearest
        neighbors.

        INPUTS:
            X: array of training data, with features as columns and observations
            in rows
            y: training data labels. Optional - if set to None it is assumed all
            data are negative instances.
            majority_label: label of data on which to train, by default set to
            -1. Data without this label will be discarded.
        RETURNS:
            None
        """
        if y is not None:
            self.train_X_ = X[y == majority_label,:].copy()
        else:
            self.train_X_ = X.copy()
        
        # Find sum of distances to nearest neighbor in mahalanobis distances
        sum_of_mahalanobis_distances = self.get_mahalanobis_distances(X = self.train_X_,
                                                                 train_mode = True,
                                                                 k = self.k)

        # set threshold value based on prediction
        self.threshold_ = np.quantile(sum_of_mahalanobis_distances, 1-self.alpha)
    
    def kneighbors(self, X: np.ndarray, n_neighbors: int = 5) -> np.ndarray:
        """Returns the sum-of-square distances to the nearest n_neighbors for
        every row in X.
        """
        if not isinstance(n_neighbors, int): n_neighbors = int(n_neighbors)
        
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
        pred_y[sum_of_mahalanobis_distances > self.threshold_] = outlier_label
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
            self.nn_.fit(X)
            K = self.K + 1
            k = self.k + 1
        else:
            K = self.K 
        
        # Find nearest samples
        nearest_neighbors = self.nn_.kneighbors(X, n_neighbors = K,
                                    return_distance = False)
        if train_mode:
            # The first entry in each row is the sample itself
            nearest_neighbors = nearest_neighbors[:,1:]
        
        
        sum_of_distances = np.zeros(X.shape[0])
        
        for i in range(X.shape[0]):
            
            # Compute precision matrix
            precision_matrix = self.precision_method_().fit(
                    self.train_X_[nearest_neighbors[i,:],:]).get_precision()
            
            if not (precision_matrix == 0).all():

                # Make assumption nearest neighbors in Euclidean space also nearest in mahalanobis space
                mahalanobis_distances = mahalanobis(sample_to_score = X[i,:],
                                                    reference_array = self.train_X_[nearest_neighbors[i,:],:],
                                                    precision_matrix = precision_matrix)
                
                # Use k-1 because first index counted as zero
                sum_of_distances[i] = np.partition(mahalanobis_distances,k-1)[0:k].sum()
        
        return sum_of_distances
        
    
    
    