#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 15:40:11 2020

@author: danielyaeger
"""
from numba import jit, njit
import numpy as np

@njit
def kNN_distances(array_to_score: np.ndarray, reference_array: np.ndarray, k: int,
                  train_mode: bool = True):
    """Finds sum-of-square distance to k nearest neighbors for samples in
    array_to_score relative to observations in reference_array. Returns an
    array of sum-of-square distances in the same order as the samples in
    array_to_score. If train mode set to True, adds 1 to k because the
    distance between two identical samples will be incidentally computed
    for every sample. 
    
    The inner loop of this algorithm should have complexity
    of O(n*d + n + 2*k) ~ O(n*d), where n is the number of rows in the reference
    array, d is the dimension of each row vector, and k is the k parameter of
    k nearest neighbors.
    """
    if train_mode: k += 1
    sum_of_square_distances = np.zeros(array_to_score.shape[0])
    for i in range(array_to_score.shape[0]):
        # Pre-allocate vector to hold distances
        distances = np.zeros(reference_array.shape[0])
 
        # numba does not support axis arguments for numpy functions. Calculate differences
        differences = array_to_score[i,:] - reference_array
        
        # Calculate norm^2
        for j in range(differences.shape[0]):
            distances[j] = np.dot(differences[j],differences[j])
        
        # take the first k indices of partitioned vector. k-1 because the first position
        # is counted as zero
        k_vec = np.partition(distances,k-1)[:k]
        
        # Add sum-of-squared distances to sample
        sum_of_square_distances[i] = np.sum(k_vec)
    return sum_of_square_distances


@njit
def KNN_samples(array_to_score: np.ndarray, reference_array: np.ndarray, K: int,
                  train_mode: bool = True):
    """Finds the nearest K observations in reference_array for each row in 
    array_to_score. Returns an array of shape:
        
        (number of obs. in reference_array, K, number of columns)
    """
    nearest_samples = np.zeros((array_to_score.shape[0],K,array_to_score.shape[1]))
    if train_mode: K += 1
    for i in range(array_to_score.shape[0]):
        # Pre-allocate vector to hold distances
        distances = np.zeros(reference_array.shape[0])
 
        # numba does not support axis arguments for numpy functions. Calculate differences
        differences = array_to_score[i,:] - reference_array
        
        # Calculate norm and distances
        for j in range(differences.shape[0]):
            distances[j] = np.linalg.norm(differences[j])
        
        # If train mode, excludes the first index
        if train_mode:
            K_indices = np.argsort(distances)[1:K]
        else:
            K_indices = np.argsort(distances)[:K]

        nearest_samples[i,:,:] = reference_array[K_indices,:]
    return nearest_samples

@njit
def knn_mahalanobis(array_to_score: np.ndarray, reference_array: np.ndarray,
                    precision_matrices: np.ndarray, k: int, 
                    train_mode: bool = True):
    """Finds the distance of the k-nearest neighbors to each sample in terms
    of the mahalanobis distance. Returns an array of sum-of-square distances 
    in the same order as the samples in array_to_score. If train mode set to 
    True, adds 1 to k because the distance between two identical samples will 
    be incidentally computed for every sample.
    """
    if train_mode: k += 1
    sum_of_square_distances = np.zeros(array_to_score.shape[0])
    for i in range(array_to_score.shape[0]):
        # Pre-allocate vector to hold distances
        distances = np.zeros(reference_array.shape[0])
 
        # numba does not support axis arguments for numpy functions. Calculate differences
        differences = array_to_score[i,:] - reference_array
        
        # Calculate norm and distances
        for j in range(differences.shape[0]):
            diff = differences[j,:]
            distances[j] = diff@precision_matrices[i,:,:]@diff.T
        
        # argpartition not supported by numba
        k_vec = np.sort(distances)[:k]
        
        # Add sum-of-squared distances to sample
        sum_of_square_distances[i] = np.sum(k_vec)
    return sum_of_square_distances




        
    