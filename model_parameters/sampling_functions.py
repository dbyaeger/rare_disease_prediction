#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 14:40:19 2020

@author: yaeger

Wrapper functions for imblearn sampling strategies. Each wrapper instantiates
the class(es), runs the resampling method, and returns the resampled array and
labels.
"""
import numpy as np
from imblearn.under_sampling import (TomekLinks, OneSidedSelection, 
                                     RandomUnderSampler)

from imblearn.over_sampling import (SMOTE, RandomOverSampler, KMeansSMOTE,
                                    ADASYN)

def tomek_links(x: np.ndarray,y: np.ndarray):
    """Returns tomek resampled x and y for which majority class Tomek Link
    members have been eliminated
    """
    tl = TomekLinks(n_jobs=4)
    return tl.fit_resample(x,y)

def one_sided_selection(x: np.ndarray,y: np.ndarray):
    """Returns tomek resampled x and y for which majority class Tomek Link
    members have been eliminated and non-informative majority class samples
    have also been culled.
    """
    oss = OneSidedSelection(n_jobs=4)
    return oss.fit_resample(x,y)
    
def random_undersample(x: np.ndarray,y: np.ndarray, sampling_strategy:float):
    """Returns array and labels for which the majority class of the input
    array has been randomly undersampled. sampling_strategy specifies the desired
    ratio of the number of samples in the minority class over the number of 
    samples in the majority class.
    """
    rus = RandomUnderSampler(sampling_strategy=sampling_strategy)
    return rus.fit_resample(x,y)

def smote(x: np.ndarray,y: np.ndarray, sampling_strategy: float, k_neighbors: int):
    """Returns array and labels for which the minority class of the input
    array has been randomly oversampled using SMOTE technique. sampling_strategy 
    specifies the desired ratio of the number of samples in the minority class 
    over the number of samples in the majority class.
    """
    if not isinstance(k_neighbors, int): k_neighbors = int(k_neighbors)
    
    sm = SMOTE(sampling_strategy=sampling_strategy)
    return sm.fit_resample(x,y)

def random_undersample_smote(x: np.ndarray,y: np.ndarray, 
                             sampling_strategy_1:float, sampling_strategy_2:float,
                             k_neighbors: int, epsilon: float = 0.01):
    """Returns array and labels for which the majority class of the input
    array has been randomly undersampled and then the minority class has been
    randomly oversampled using SMOTE technique. sampling_strategy_1
    specifies the desired ratio of the number of samples in the minority class 
    over the number of samples in the majority class after random_undersampling,
    and sampling_strategy_2 specifies the ratio after SMOTE. In case 
    sampling_strategy_2 <= sampling_strategy_1, sampling_strategy_2 will be changed
    to sampling_strategy_1 + epsilon
    """
    if (sampling_strategy_2 <= sampling_strategy_1) and sampling_strategy_1 <= 0.99:
        sampling_strategy_2 = sampling_strategy_1 + epsilon
    elif (sampling_strategy_2 <= sampling_strategy_1) and sampling_strategy_1 > 0.99:
        sampling_strategy_2 = 1.0
    
    if not isinstance(k_neighbors, int): k_neighbors = int(k_neighbors)
    
    rus = RandomUnderSampler(sampling_strategy=sampling_strategy_1)
    x, y = rus.fit_resample(x,y)
    if sampling_strategy_1 <= 0.99:
        sm = SMOTE(sampling_strategy=sampling_strategy_2, k_neighbors=k_neighbors)
    return sm.fit_resample(x,y)

def random_oversample(x: np.ndarray,y: np.ndarray, sampling_strategy:float):
    """Returns array and labels for which the minority class of the input
    array has been randomly oversampled. sampling_strategy specifies the desired
    ratio of the number of samples in the minority class over the number of 
    samples in the majority class.
    """
    ros = RandomOverSampler(sampling_strategy=sampling_strategy)
    return ros.fit_resample(x, y)

def kmeans_smote(x: np.ndarray,y: np.ndarray, sampling_strategy: float, 
                 k_neighbors: int, cluster_balance_threshold: float):
    """Returns arrays for which the minority class has been oversampled using
    KMeansSmote. Sampling_strategy is the desired ratio of minority to majority
    samples, k_neighbors are the number of neighbors to use when creating new
    synthetic minority samples, and cluster_balance_threshold is the balance
    ratio threshold for creating new minority samples from each cluster.
    """
    if not isinstance(k_neighbors, int): k_neighbors = int(k_neighbors)
    
    kmm = KMeansSMOTE(sampling_strategy = sampling_strategy, 
                      k_neighbors = k_neighbors, 
                      cluster_balance_threshold = cluster_balance_threshold,
                      n_jobs = 4)
    return kmm.fit_resample(x,y)

def kmeans_adasyn(x: np.ndarray,y: np.ndarray, sampling_strategy: float, 
                 n_neighbors: int):
    """Returns array for which the minority class has been oversampled using
    ADASYN. Sampling_strategy is the desired ratio of minority to majority
    samples, n_neighbors are the number of neighbors to use when creating new
    synthetic minority samples.
    """
     if not isinstance(n_neighbors, int): n_neighbors = int(n_neighbors)
     
     ada = ADASYN(sampling_strategy = sampling_strategy, n_neighbors = n_neighbors,
                  n_jobs = 4)
     return ada.fit_resample(x,y)
     
     
    