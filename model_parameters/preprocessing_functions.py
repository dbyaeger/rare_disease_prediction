#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 13:39:35 2020

@author: yaeger

Wrapper functions for principal components, partial squares, and other
pre-processing functions. Each wrapper instantiates the class(es), fits on the 
original data, and returns the transformed array and labels.

NOTE: These methods should all accept y as an argument, even if y is not
used by the method.

"""

from sklearn.decomposition import PCA, KernelPCA

def linear_pca(x, y, n_components):
    """Wrapper method for sklearn.decomposition.PCA. Fits the PCA using n_components
    number of components. Returns the transformed version of x."""
    # Ensure argument is an int
    if not isinstance(n_components, int): n_components = int(n_components)
    
    pca = PCA(n_components = n_components)
    pca.fit(x)
    return pca.fit_transform(x)

def radial_pca(x, y, n_components, gamma):
    """Wrapper method for sklearn.decomposition.PCA. Fits the PCA using n_components
    number of components. Returns the transformed version of x."""
    # Ensure argument is an int
    if not isinstance(n_components, int): n_components = int(n_components)
    
    # Set eigen_solver to arpack when n_components much less than n_samples
    pca = KernelPCA(n_components = n_components, gamma = gamma, kernel="rbf", 
                    eigen_solver = "arpack")
    pca.fit(x)
    return pca.fit_transform(x)