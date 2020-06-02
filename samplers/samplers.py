#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  8 08:39:32 2020

@author: yaeger
"""

from imblearn.under_sampling import RandomUnderSampler
import numpy as np
from collections import Counter

class ReturnMajority(RandomUnderSampler):
    """Wrapper for BaseUnderSampler class but overloads fit_resample method
    to just return majority instances (label = -1)"""
    def __init__(self):
        super().__init__()
        
    def fit_resample(self, X, y):
        """Returns the majority instances based on labels in y.
        
        INPUTS:
            X: np.ndarray with observations in rows and features in columns
            y: np.ndarray of labels
        
        Returns:
            X: np.ndarray with rows corresponding to majority labels
            y: np.ndarray with only majority label
        """
        #Find minority and majority labels
        label_key_counts = [(k,v) for (k,v) in Counter(list(y)).items()]
        sorted_label_tuples = sorted(label_key_counts, key = lambda x: x[1])
        majority_label = sorted_label_tuples[-1][0]
        
        return X[y == majority_label,:], y[y == majority_label]

class ReturnMinority(RandomUnderSampler):
    """Wrapper for BaseUnderSampler class but overloads fit_resample method
    to just return majority instances (label = -1)"""
    def __init__(self):
        super().__init__()
        
    def fit_resample(self, X, y):
        """Returns the minority instances based on labels in y.
        
        INPUTS:
            X: np.ndarray with observations in rows and features in columns
            y: np.ndarray of labels
        
        Returns:
            X: np.ndarray with rows corresponding to minority labels
            y: np.ndarray with only minority label
        """
        #Find minority and majority labels
        label_key_counts = [(k,v) for (k,v) in Counter(list(y)).items()]
        sorted_label_tuples = sorted(label_key_counts, key = lambda x: x[1])
        minority_label = sorted_label_tuples[0][0]
        
        return X[y == minority_label,:], y[y == minority_label]
        
            