#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 16:13:38 2020

@author: yaeger
"""
from imblearn.metrics import geometric_mean_score
from sklearn.metrics import make_scorer
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from shrink.shrink import SHRINK
shrink = SHRINK(T=10,metric = make_scorer(geometric_mean_score), theta = 4)
from sklearn.datasets import load_iris
data = load_iris()
X = data['data']
y = data['target'] 
y[y > 0] = -1 
y[y == 0] = 1 
shrink.fit(X,y) 

def test_on_iris():
    """Loads the iris dataset and evaluates SHRINK algorithm on iris dataset 
    """
    # Get data
    data = load_iris()
    X = data['data']
    y = data['target']
    metric = make_scorer(geometric_mean_score)
    
    # Reframe as a 2-class problem
    y[y > 0] = -1 
    y[y == 0] = 1 
    
    # Train shrink algorithm
    shrink = SHRINK(T=10, metric = metric, theta = 2)
    shrink.fit(X,y) 
    
    #
    
    