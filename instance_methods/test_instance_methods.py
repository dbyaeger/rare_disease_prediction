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
import matplotlib.pyplot as plt

# Code for testing fault detection KNN
def get_fdKNN_distances(X: np.ndarray, y: np.ndarray, k: int = 5):
    """Wrapper for kneighbors function of FaultDetectionKNN"""
    fdKNN = FaultDetectionKNN(k = k, alpha = 0.05)
    fdKNN.fit(X,y)
    return fdKNN.kneighbors(X, n_neighbors = k)

    
def get_skKNN_distances(X: np.ndarray, y: np.ndarray, k: int = 5):
    """Wrapper for kneighbors function of sklearn's NearestNeighbors. Returns
    the squared distances"""
    skKNN = NearestNeighbors(n_neighbors = k, metric = 'euclidean')
    skKNN.fit(X,y)
    dist_skKNN = skKNN.kneighbors(X, n_neighbors = k)[0]
    return (dist_skKNN*dist_skKNN).sum(axis=1)

def test_fault_detection_knn(X: np.ndarray, k: int = 5):
    """Tests the FaultDetectionKNN by comparing its results to the sklearn
    NearestNeighbors algorithm. Returns True if test passed and false otherwise.
    """
    # Create fake y-vector
    y = np.ones(X.shape[0])*(-1)
    
    return np.allclose(get_fdKNN_distances(X,y,k), get_skKNN_distances(X,y,k))

# Code for testing MahalanobisDistanceKNN

def make_data(N_samples: int = 2000):
    """Code for generating heart-shaped data as in Verdier and Ferreira.
    Returns array with 2D heart-shaped distribution.
    """
    mu1, cov1 = np.array([3,1]), np.array([[0.7,0.5],[0.5,0.7]])
    mu2, cov2 = np.array([0,1]), np.array([[0.7,-0.5],[-0.5,0.7]])
    N1 = np.random.multivariate_normal(mu1,cov1,size=N_samples//2)
    N2 = np.random.multivariate_normal(mu2,cov2,size=N_samples//2)
    return np.concatenate((N1,N2))

def train_and_test_mahalanobis_distance_knn(X: np.ndarray, y: np.ndarray):
    """Returns 2D array of predicted labels and coordinates using 
    mahalanobis_distance_knn trained on the array X.
    """
    knn = MahalanobisDistanceKNN(K = 150, k = 20, alpha = 0.01)
    knn.fit(X,y)
    #TODO - finish
    
        

