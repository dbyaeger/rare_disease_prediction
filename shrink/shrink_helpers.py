#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 13:23:03 2020

@author: yaeger
"""
import numpy as np
from sklearn.base import BaseEstimator

class RangeClassifier(BaseEstimator):
    """Very simple classifier that returns 1 for each entry that is within
    the range [min, max] and false otherwise.
    
    INPUTS:
        X: 1-D numpy array of training data
        minority_label: what value denotes the minority label (default of 1)
        majority_label: what value denotes the majority label (default of -1)
        
    """
    def  __init__(self, X: np.ndarray, majority_label: int = -1, minority_label: int = 1):
        assert len(X) >= 2, f'X: {X} has length less than 2!'
        
        # Sort array if not already sorted
        if not np.all(X[:-1] <= X[1:]):
            X.sort()
            
        self.X = X
        self.min = X[0]
        self.max = X[-1]
        
        assert self.min < self.max, 'minimum greater than or equal to maximum!'
        self.majority_label = majority_label
        self.minority_label = minority_label
        
    def decision_function(self,X):
        """Given an input vector X, returns an array with 1 for each entry of X 
        within the range [min, max] and 0 otherwise, where min and max are the 
        attribute values of minimum and maximum established at instantiation.
        """
        return np.array(list(map(lambda x, minx = self.min, maxx = self.max: int(minx <= x <= maxx), X)))
    
    def predict(self, X):
        """Given an input vector X, returns an array with minority_label for 
        each entry of X within the range [min, max] and majority_label otherwise, 
        where min and max are the attribute values of minimum and maximum 
        established at instantiation, and majority_label and minority_label are
        also attributes established at instantiation.
        """
        decisions = self.decision_function(X)
        decisions[decisions == 0] = self.majority_label
        decisions[decisions == 1] = self.minority_label
        return decisions
    
    def feature_values(self):
        """Returns the sorted X array"""
        return self.X

class RangeClassifierHandler():
    """Implements a single weak classifier of type used in the SHRINK algorithm
    from "Machine Learning for the Detection of Oil Spills in Satellite Radar 
    Images" by Kubat, Holte, and Matwin. Trains a series RangeClassifiers, and
    uses only the best-performing version on the metric for prediction. The
    prediction of the best-performing RangeClassifier in training is weighted
    by its performance on the training set using the metric.
    
    INPUTS:
        train_feature: a single feaute column from the training data as a
            1-D numpy array
        train_labels: training labels as a 1-D numpy array
        metric: the metric to be used in weighting and assessing classifiers.
            Should take the classifier, data, and label as input and return
            a float.
        minority_label: what value denotes the minority label (default of 1)
        majority_label: what value denotes the majority label (default of -1)
    
    PARAMETERS: 
        keep_all_classifiers: if set to True, all classifiers and weights
            created during training will be saved. Otherwise, only the best
            classifer and corresponding best weight will be saved.
    """
    def __init__(self, train_feature: np.ndarray, train_labels: np.ndarray, 
             metric: callable, feature: int, minority_label: int = -1,
             majority_label: int = 1, keep_all_classifiers: bool = True):
        
        assert len(train_feature.shape) == 1, f'train feature should be a 1-D array, not an array with shape {train_feature.shape}'
        self.eval_train_feature = train_feature
        self.eval_train_labels = train_labels
        
        # np.unique returns a sorted array
        feature_values = np.unique(train_feature[train_labels == minority_label])
        #assert len(feature_values) > 2, f"{feature_values} doesn't have enough unique values!"
        self.minority_label = minority_label
        self.majority_label = majority_label
        self.metric = metric
        self.feature = feature
        self.keep_all_classifiers = keep_all_classifiers
        
        #start with base classifier with entire range
        classifier, score = self._create_and_score_classifier(feature_values)
        self.classifiers = [classifier]
        self.weights = [score]
    
    def _create_and_score_classifier(self, X: np.ndarray):
        """Instantiates a new classifier with the array X and evaluates the 
        classifier using the metric attribute. Returns the classifier and the
        score on the metric.
        """
        classifier = RangeClassifier(X = X, majority_label = self.majority_label,
                                     minority_label = self.minority_label)
        score = self.metric(classifier, self.eval_train_feature, self.eval_train_labels)
        return classifier, score
    
    def train(self):
        """On each iteration create a new classifier by moving either left or 
        right endpoint in by one value, depending on which acheives a higher 
        value on the metric. Also calculate weight for new classifier"""
        # Get feature values from last trained classifer. feature_values is sorted 
        feature_values = self.classifiers[-1].feature_values()
        
        # if feature values has length of 3, only 2 additional classifers can be trained
        if len(feature_values) >= 3:
            left_classifier, left_score = self._create_and_score_classifier(feature_values[1:])
            right_classifier, right_score = self._create_and_score_classifier(feature_values[:-1])
            
            # Keep the classifier with the best score
            if left_score < right_score:
                self.classifiers.append(left_classifier)
                self.weights.append(left_score)
            else:
                self.classifiers.append(right_classifier)
                self.weights.append(right_score)

    def prune(self):
        """Calling the prune method causes only the best classifier to be 
        retained amongst all the trained classifiers.
        """
        # Get best classifier and weight
        best_classifier = self.classifiers[np.argmax(self.weights)]
        best_weight = np.max(self.weights)
        
        # delete old values
        if not self.keep_all_classifiers:
            del self.classifiers
            del self.weights
        
        # Only save the best values
        self.classifier = best_classifier
        self.weight = best_weight
        
    def decision_function(self,X):
        """Returns the decision function of the best classifier on the training
        set weighted by the value of the score metric for that classifier on
        the training set for each value of the input array X.
        """
        return self.weight*self.classifier.decision_function(X)
    
    def get_weight(self):
        """Returns the weight attribute"""
        return self.weight
    
    def set_weight_to_zero(self):
        """Sets the weight attribute to zero"""
        self.weight = 0
    
        