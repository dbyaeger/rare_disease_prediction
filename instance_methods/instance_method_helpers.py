#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 15:40:11 2020

@author: danielyaeger
"""
from numba import njit
import numpy as np


@njit
def mahalanobis(sample_to_score: np.ndarray, reference_array: np.ndarray,
                precision_matrix: np.ndarray):
    """Finds the distance of each row in a reference array from a sample vector. 
    Returns a vector of mahalanobis distances.
    """
    differences = np.zeros(reference_array.shape[0])
    for i in range(reference_array.shape[0]):
        delta = sample_to_score - reference_array[i,:]
        differences[i] = delta.T@precision_matrix@delta
    return differences




        
    