#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 16:05:32 2020

@author: yaeger
"""

def proportion_correctly_ranked(y_pred, y_true_labels,
                                label_map = {'Likely': 1,'Possible': 2,
                                            'Unlikely': 3}):
    """Yields a metric similar to MannWhitney U but that is created for cases
    in which y_pred is an interval measure and y_true is ordinal. Calculates the
    percentage of correctly ranked pairs. Only considers pairs with differently
    ranked true catgeorical labels.

    INPUTS:
        y_pred: array or list of rankable numeric values. It is assumed that lower
        scores should be ranked ahead of higher scores (i.e. lower is better).

        y_true: list of categories (which are represented as strings, i.e

                    ['Unlikely', 'Possible', 'Possible', 'Likely', ...]

                Entry i in y_true corresponds to the true categorical ranking of
                entry i in y_pred

        label_map: a dictionary in which each category in y_true is a key and
        the value is the ordinal rank of that category, e.g.:

                    {'Likely': 1,'Possible': 2, 'Unlikely': 3}

    Returns the total proportion of correctly ranked pairs.
    """
    # Check to make sure that all classes in label_map are in y_true_labels
    for key in label_map:
        assert key in y_true_labels, f'Class {key} in label_map but not in y_true_labels!'

    # Enumerate the total number of possible pairs
    possible_correct = 0
    for key_1 in label_map:
        for key_2 in label_map:
            if label_map[key_1] < label_map[key_2]:
                possible_correct += y_true_labels.count(key_1)*y_true_labels.count(key_2)

    # Find the total number of correctly ranked pairs
    actual_correct = 0
    for i in range(len(y_pred)-1):
        for j in range(i,len(y_pred)):
            if label_map[y_true_labels[i]] <  label_map[y_true_labels[j]]:
                if y_pred[i] < y_pred[j]: actual_correct += 1
            elif label_map[y_true_labels[i]] > label_map[y_true_labels[j]]:
                if y_pred[i] > y_pred[j]: actual_correct += 1

    return actual_correct/possible_correct

def convert_labels_to_integers(y_true_labels, label_map = {'Likely': 1,
                                                             'Possible': 1,
                                                             'Unlikely': -1}):
    """ Takes in a list of categorical labels (which are represented as strings)
    and a label_map, which contains the mapping from categories to integers, and
    returns list of integers as labels.
    """
    # Check to make sure that all classes in y_true_labels are in label_map
    for category in y_true_labels:
        assert category in label_map, f'Class {category} in in y_true_labels but not in label_map!'

    integer_labels = list(map(lambda x, label_map = label_map: label_map[x], \
                         y_true_labels))
    return integer_labels
