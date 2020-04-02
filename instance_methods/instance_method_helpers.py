#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 15:40:11 2020

@author: danielyaeger
"""
from numba import jit, njit
import numpy as np
from sklearn.covariance import LedoitWolf, MinCovDet

@njit
def update_kvec(k_vec: np.ndarray, dist: float, k: int):
        """Updates the array k_vec to include the distances to the k nearest
        neighbors based on the new distance dist. Returns k_vec."""
        # if distance greater than final entry, return array
        if dist > k_vec[-1]:
            return k_vec
        elif dist < k_vec[-1]:
            index = np.searchsorted(k_vec,dist)
            if index == (k-1):
                k_vec[-1] = dist
            else:
                k_vec[index:] = np.roll(k_vec[index:],1)
                k_vec[index] = dist
        return k_vec
    
@njit
def update_Kvec_Kindices(K_vec: np.ndarray, K_indices: np.ndarray, dist: float, 
                new_neighbor: int, K: int):
        """Updates the array K_vec and the array K_indices to include the 
        distances to the K nearest neighbors and the indices of the closest
        K neighbors based on the new distance dist from neighbor new_neighbor. 
        Returns K_vec and K_indices."""
        # if distance greater than final entry, return array
        if dist > K_vec[-1]:
            return K_vec, K_indices
        elif dist < K_vec[-1]:
            index = np.searchsorted(K_vec,dist)
            if index == (K-1):
                K_vec[-1] = dist
                K_indices[-1] = new_neighbor
            else:
                K_vec[index:] = np.roll(K_vec[index:],1)
                K_vec[index] = dist
                K_indices = np.roll(K_indices,1)
                K_indices[index] = new_neighbor
        return K_vec, K_indices

@njit
def kNN_distances(array_to_score: np.ndarray, reference_array: np.ndarray, k: int,
                  train_mode: bool = True):
    """Finds sum-of-square distance to k nearest neighbors for samples in
    array_to_score relative to observations in reference_array. Returns an
    array of sum-of-square distances in the same order as the samples in
    array_to_score. If train mode set to True, adds 1 to k because the
    distance between two identical samples will be incidentally computed
    for every sample.
    """
    if train_mode: k += 1
    sum_of_square_distances = np.zeros(array_to_score.shape[0])
    for i in range(array_to_score.shape[0]):
        sample = array_to_score[i,:]
        k_vec = np.ones(k)*np.inf
        for row in reference_array:
            # Update k vector based on distance to training observation
            k_vec = update_kvec(k_vec = k_vec, dist = np.linalg.norm(sample-row),
                                k = k)
        # Add sum-of-squared distances to sample
        sum_of_square_distances[i] = np.dot(k_vec,k_vec)
    return sum_of_square_distances

@njit
def KNN_precision(array_to_score: np.ndarray, reference_array: np.ndarray, K: int,
                  train_mode: bool = True):
    """Finds the precision matrix using the nearest K samples in Euclidean space
    for each sample. Returns an array of precision matrices in which the ith 
    precision matrix corresponds to the ith sample. Estimates covariance using
    the LedoitWolf method."""
    precision_matrices = np.zeros(array_to_score.shape[0],array_to_score.shape[1],array_to_score.shape[1])
    for i in range(array_to_score.shape[0]):
        sample = array_to_score[i,:]
        K_vec = np.ones(K)*np.inf
        K_indices = np.ones(K)*np.inf
        for j in range(reference_array.shape[0]):
                row = reference_array[j,:]
                if train_mode:
                    if i == j:
                        if np.all(row == sample): continue
                K_vec, K_indices = update_Kvec_Kindices(K_vec = K_vec, 
                                   K_indices = K_indices,
                                   dist = np.linalg.norm(sample-row),
                                   new_neighbor = j, K = K)
        K_Nearest_Samples = reference_array[K_indices,:]
        precision_matrices[i] = LedoitWolf().fit(K_Nearest_Samples).precision_
    return precision_matrices

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
    sum_of_k_distances = np.zeros(array_to_score.shape[0])
    for i in range(array_to_score.shape[0]):
        sample = array_to_score[i,:]
        k_vec = np.ones(k)*np.inf
        for row in reference_array:
            # Calculate the Mahalanobis distance. May be faster with njit than sklearn method
            dist = (sample - row).transpose@precision_matrices[i,:,:]@(sample - row)
            # Update k vector based on distance to training observation
            k_vec = update_kvec(k_vec = k_vec, dist = dist, k = k)
        # Add sum-of-squared distances to sample
        sum_of_k_distances[i] = k_vec.sum()
    return sum_of_k_distances




        
    