#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 11:14:35 2020

@author: yaeger

Collection of test function to test the Instance Methods by comparing the
results to sklearn functions
"""
import numpy as np
from sklearn.neighbors import NearestNeighbors
from instance_methods.InstanceMethods import FaultDetectionKNN, MahalanobisDistanceKNN

def get_fdKNN_distances(X: np.ndarray, y: np.ndarray, k: int = 5):
    """Wrapper for kneighbors function of FaultDetectionKNN"""
    fdKNN = FaultDetectionKNN(k = k, alpha = 0.05)
    fdKNN.fit(X,y)
    return fdKNN.kneighbors(X, n_neighbors = k)

def get_mdKNN_distances(X: np.ndarray, y: np.ndarray, K: int = 100, k: int = 5):
    """Wrapper for kneighbors function of MahalanobisDistanceKNN"""
    mdKNN = MahalanobisDistanceKNN(K= K, k = k, alpha = 0.05)
    mdKNN.fit(X,y)
    return mdKNN.kneighbors(X, n_neighbors = k)
    
def get_skKNN_distances(X: np.ndarray, y: np.ndarray, k: int = 5):
    """Wrapper for kneighbors function of sklearn's NearestNeighbors. Returns
    the squared distances"""
    skKNN = NearestNeighbors(n_neighbors = k, metric = 'euclidean')
    skKNN.fit(X,y)
    dist_skKNN = skKNN.kneighbors(X, n_neighbors = k)[0]
    return (dist_skKNN*dist_skKNN).sum(axis=1)

def wrapper2():
    X = np.random.random((10000,10)) 
    y = np.ones(10000)*(-1)
    return get_skKNN_distances(X,y,20)


def test_fault_detection_knn(X: np.ndarray, k: int = 5):
    """Tests the FaultDetectionKNN by comparing its results to the sklearn
    NearestNeighbors algorithm. Returns True if test passed and false otherwise.
    """
    # Create fake y-vector
    y = np.ones(X.shape[0])*(-1)
    
    return np.allclose(get_fdKNN_distances(X,y,k), get_skKNN_distances(X,y,k))

def test_mahalanobis_distance_knn(X,np.ndarray, K: int = 100 k: int = 5):
    """Test the MahalanobisDistanceKNN by comparing its results to the sklearn
    NearestNeighbors algorithm. Returns True if test passed and false otherwise.
    """
    # Create fake y-vector
    y = np.ones(X.shape[0])*(-1)
    
    mdKNN_distances = get_mdKNN_distances(X,K,k)
    
    
        

