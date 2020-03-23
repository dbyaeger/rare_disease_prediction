#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 15:40:11 2020

@author: danielyaeger
"""
from numba import njit
import numpy as np

@njit
def update_k(k: np.ndarray, dist: float):
        """Updates the array k to include the distances to the k nearest
        neighbors based on the new distance dist. Returns k."""
        # if distance greater than final entry, return array
        if dist > k[-1]:
            return k
        elif dist < k[0]:
            k = np.roll(k,1)
            k[0] = dist
        elif k[0] < dist < k[-1]:
            index = np.searchsorted(k,dist)
            if index == (len(k)-1):
                k[-1] = dist
            else:
                k[index:] = np.roll(k[index:],1)
                k[index] = dist
        return k