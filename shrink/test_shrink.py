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
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import numpy as np

def test_on_shrink_iris():
    """Loads the iris dataset and evaluates SHRINK algorithm on iris dataset 
    """
    # Get data
    data = load_iris()
    X = data['data']
    
    # Run PCA to reduce to 2-class problem
    pca = PCA(n_components = 2)
    pca.fit(X)
    X = pca.fit_transform(X)
    y = data['target']
    metric = make_scorer(geometric_mean_score)
    
    # Reframe as a 2-class problem
    y[y > 0] = -1 
    y[y == 0] = 1 
    
    # Train shrink algorithm
    shrink = SHRINK(T=100, metric = metric, theta = 0.9)
    shrink.fit(X,y) 
    
    # Evaluate shrink algorithm
    x1_min, x1_max = X[:,0].min() - 1, X[:,0].max() + 1
    x2_min, x2_max = X[:,1].min() - 1, X[:,1].max() + 1
    x1, x2 = np.meshgrid(np.arange(x1_min, x1_max, 0.1),
                     np.arange(x2_min, x2_max, 0.1))
    coords = np.c_[x1.ravel(), x2.ravel()]
    
    # Get predictions
    predictions = shrink.predict(coords)
    
    #Plot
    plt.scatter(X[:,0],X[:,1], s=20, edgecolor='k')
    plt.contourf(x1,x2,predictions.reshape(x1.shape),alpha=0.4)
    plt.title('SHRINK classifier decision boundaries on iris data')
    plt.show()
    
if __name__ == "__main__":
    test_on_shrink_iris()