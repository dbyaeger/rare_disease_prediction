#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 16:05:32 2020

@author: yaeger
"""

def proportion_correctly_ranked(y_pred, y_true_labels, 
                                label_map = {'Likely': 1,'Possible': 2,
                                            'Unlikely': 3}):
    """Yields a metric similar to MannWhitney U but that is compatible with.
    cases for which y_pred is an interval measure and y_true is ordinal. Assumes
    the lowest scores should be labeled as class 1, and the highest scores as
    class n.
    
    Returns the total proportion of correctly ranked pairs. 
    """
    actual_correct = 0
    possible_correct = 0
    num_obs = len(y_pred)
    for i in range(num_obs-1):
        for j in range(i,num_obs):
            if label_map[y_true_labels[i]] <  label_map[y_true_labels[j]]:
                possible_correct += 1
                if y_pred[i] < y_pred[j]: actual_correct += 1
            elif label_map[y_true_labels[i]] > label_map[y_true_labels[j]]:
                possible_correct += 1
                if y_pred[i] > y_pred[j]: actual_correct += 1
    return actual_correct/possible_correct
                