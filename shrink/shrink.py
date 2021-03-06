#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 11:39:48 2020

@author: yaeger
"""
import numpy as np
from collections import Counter
from shrink.shrink_helpers import RangeClassifierHandler
from imblearn.metrics import geometric_mean_score
from sklearn.metrics import make_scorer
from sklearn.base import BaseEstimator

class SHRINK(BaseEstimator):
    """Implementation of SHRINK algorithm from "Machine Learning for the Detection
    of Oil Spills in Satellite Radar Images" by Kubat, Holte, and Matwin. Trains
    an ensemble of classifiers for each feature of the form:
        
        1 if x in [min,max], otherwise 0
        
        Each classifier is weighted by its performance on the performance metric,
        so the decision function is of the form:
            
            Sum over i classifier_i(x) X weight_i
        
        And observations are predicted to belong to the minority class if the
        decision function is greater or equal to theta
        
        PARAMETERS:
            T: the number of iterations to use in training the simple classifiers.
                On each iteration, either the right or left endpoint is removed.
            metric: the metric to use when training the classifier. The metric
                should take the classifier, data, and true labels as input.
            theta: the threshold to use in deciding whether an observation
                belongs to the the minority or majority class.
            metric_performance_threshold: the value a potential classifier 
                should have on the metric for it to be included in the final 
                classifier.
            
    """
    def __init__(self, T: int = 10, theta: float = 5, metric: callable = make_scorer(geometric_mean_score),
                 metric_performance_threshold: float = 0.5):
        
        assert metric_performance_threshold < 1, "Metric performance threshold must be less than one"
        
        self.T = T
        self.metric = metric
        self.theta = theta
        self.metric_performance_threshold = metric_performance_threshold
    
    def fit(self, X:np.ndarray, y: np.ndarray):
        """Fits SHRINK method to data.
        
        INPUTS:
            X: numpy array of training data
            y: numpy array of training labels
        
        Returns None
        """
        #Find minority and majority labels
        label_key_counts = [(k,v) for (k,v) in Counter(list(y)).items()]
        sorted_label_tuples = sorted(label_key_counts, key = lambda x: x[1])
        minority_label = sorted_label_tuples[0][0]
        majority_label = sorted_label_tuples[-1][0]
                
        #Instantiate labelers
        self.classifiers_ = []
        for feature in range(X.shape[1]):
            train_feature = X[:,feature]
            # Sometimes, especially with cross-validation, only 1 unique feature value
            if len(np.unique(train_feature[y == minority_label])) >= 2:
                self.classifiers_.append(RangeClassifierHandler(train_feature = train_feature, 
                                      train_labels = y,
                                      minority_label = minority_label,
                                      majority_label = majority_label,
                                      metric =self.metric,
                                      feature = feature))
        
        #Train. First training for each classifier happens on instantiation
        for t in range(self.T-1):
            for feature in range(len(self.classifiers_)):
                self.classifiers_[feature].train()
                
        #Select best classifiers_
        for feature in range(len(self.classifiers_)):
            self.classifiers_[feature].prune()
            
        #If classifier weight less than metric_performance_threshold, set classifier weight to zero
        for classifier in self.classifiers_:
            if classifier.weight < self.metric_performance_threshold:
                classifier.set_weight_to_zero()
                
    def predict(self, X: np.ndarray, minority_label = 1, majority_label = -1):
        """Returns predicted class based on input array.

        INPUTS:
            X: data array, with features as columns and observations
            in rows.
            outlier_label: label for abnormal samples. Default of 1
            majority_label: label for normal samples. Default of -1
        RETURNS:
            pred_y: array of predicted classes
        """
        predictions = np.ones(X.shape[0])*majority_label
        
        # Sum up predictions of classifiers_
        decision_sums = self.decision_function(X)
        
        # Where predictions >= lambda, set label to minority label
        predictions[decision_sums >= self.theta] = minority_label
        
        return predictions
        
    def decision_function(self, X: np.ndarray):
        """Returns decision function for input array.

        INPUTS:
            X: data array, with features as columns and observations
            in rows.
        RETURNS:
            decision function value for each observation in X.
        """
        # Sum up predictions of classifiers_
        return np.array([clf.decision_function(X[:,clf.feature]) for clf in self.classifiers_]).sum(axis = 0)
     
        
        
        
            
        
        
        
        