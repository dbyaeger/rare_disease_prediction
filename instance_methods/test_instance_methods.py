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

def train_and_test_classifier(X: np.ndarray, classifier: callable, **args):
    """Returns 2D array of predicted labels and coordinates using 
    mahalanobis_distance_knn trained on the array X.
    """
    knn = classifier(**args)
    knn.fit(X)
    
    # Generate data for testing
    x1_min, x1_max = X[:,0].min() - 1, X[:,0].max() + 1
    x2_min, x2_max = X[:,1].min() - 1, X[:,1].max() + 1
    x1, x2 = np.meshgrid(np.arange(x1_min, x1_max, 0.1),
                     np.arange(x2_min, x2_max, 0.1))
    coords = np.c_[x1.ravel(), x2.ravel()]
    # Get predictions
    predictions = knn.predict(coords)
    return x1,x2,predictions

def plot_decision_boundary():
    """Wrapper to make heart-shaped data, train classifer, and plot decision
    boundary"""
    X = make_data(1000)
    x1,x2,fd_predictions = train_and_test_classifier(X,FaultDetectionKNN,k=3,alpha=0.01,n_jobs=4)
    x1,x2,md_predictions = train_and_test_classifier(X,MahalanobisDistanceKNN,K=150,k=3,alpha=0.01,n_jobs=4)
    
    f, axarr = plt.subplots(2, 1, sharex='col', figsize=(10, 8))
    
    for idx, pred, title in zip([0,1], [fd_predictions,md_predictions], 
                                ['FaultDetectionkNN','MahalanobisDistanceKNN']):
        axarr[idx].contourf(x1,x2,pred.reshape(x1.shape),alpha=0.4)
        axarr[idx].scatter(X[:,0],X[:,1], s=20, edgecolor='k')
        axarr[idx].set_title(title)
    plt.show()
    
        
if __name__ == "__main__":
    plot_decision_boundary()
