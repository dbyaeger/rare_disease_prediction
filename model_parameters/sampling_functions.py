#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 14:40:19 2020

@author: yaeger

Wrapper functions for imblearn sampling strategies. Each wrapper instantiates
the class and returns the resampling method
"""
import numpy as np
from imblearn.under_sampling import TomekLinks, OneSidedSelection, RandomUnderSampler

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
    oss.fit_resample(x,y)
    
def random_undersample(x: np.ndarray,y: np.ndarray, sampling_strategy:float):
    """Returns array and labels for which the majority class of the input
    array has been randomly undersampled. sampling_strategy specifies the desired
    ratio of the number of samples in the minority class over the number of 
    samples in the majority class.
    """
    rus = RandomUnderSampler(sampling_strategy=sampling_strategy)
    rus.fit_resample(x,y)