#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 15:40:11 2020

@author: danielyaeger
"""
from numba import njit
import numpy as np

@njit
def update_kvec(k_vec: np.ndarray, dist: float, k: int):
        """Updates the array k_vec to include the distances to the k nearest
        neighbors based on the new distance dist. Returns k_vec."""
        # if distance greater than final entry, return array
        if dist > k_vec[-1]:
            return k_vec
        elif dist < k_vec[-1]:
            index = np.searchsorted(k_vec,dist)
            if index == (k-1):
                k_vec[-1] = dist
            else:
                k_vec[index:] = np.roll(k_vec[index:],1)
                k_vec[index] = dist
        return k_vec

@njit
def kNN_distances(array_to_score: np.ndarray, reference_array: np.ndarray, k: int):
        """Finds sum-of-square distance to k nearest neighbors for samples in
        array_to_score relative to observations in reference_array. Returns an
        array of sum-of-square distances in the same order as the samples in
        array_to_score. Ignores identical rows with the same row index when
        calculating the sum-of-square distance.
        """
        sum_of_square_distances = np.zeros(array_to_score.shape[0])
        for i, sample in enumerate(array_to_score):
            k_vec = np.ones(k)*np.inf
            for j, row in enumerate(reference_array):
                # If i==j and rows are identical, skip the row
                if i == j:
                    if (row == sample).all(): continue

                # Update k vector based on distance to training observation
                k_vec = update_kvec(k_vec = k_vec, dist = np.linalg.norm(sample-row),
                                    k = k)
            # Add sum-of-squared distances to sample
            sum_of_square_distances[i] = np.dot(k_vec,k_vec)
        return sum_of_square_distances
