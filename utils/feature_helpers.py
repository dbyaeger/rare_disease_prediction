#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 14:55:53 2020

@author: yaeger
"""
import numpy as np
import pandas as pd

def safe_log(x: np.array):
    """
    Returns the log of x. If x contains a zero value, adds 1 to all values
    """
    assert type(x) == np.ndarray, f'x is of type {type(x)}'
    
    if x.all():
        y = np.log(x)
    else:
        y = np.log(x + 1)
    
    assert not np.isinf(y).any(), f'Infinite values found in y!'
    
    return y
    

def normalize(feature: np.array):
    """
    Normalizes column
    """
    assert type(feature) == np.ndarray, f'feature is of type {type(feature)}'
    
    y = (feature - feature.min())/(feature.max() - feature.min())
    
    assert (not np.isinf(y).any()) and (not np.isnan(y).any()), f'Infinite values found in y!'
    
    assert (y.min() >= 0) and (y.max() <= 1), f"y has max {y.max()} and min {y.min()}"
    
    return y


def log_and_normalize_features(dataframe: pd.DataFrame):
    """
    Takes the log of each feature column, adding a 1 if necessary, and then normalizes.
    Returns the transformed data frame.
    
    Assumes only numeric input.
    """
   
    
    transformed_df = pd.DataFrame()
    
    for feature in dataframe.columns:
        transformed_df[feature] = normalize(safe_log(dataframe[feature].to_numpy()))
    
    return transformed_df
    