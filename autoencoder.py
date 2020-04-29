#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 15:39:06 2020

@author: yaeger
"""
from sklearn.base import BaseEstimator
from tensorflow import keras




class AutoEncoder(BaseEstimator):
    """Constructs an autoencoder for dimensionality reduction
    """