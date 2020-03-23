#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 10:24:17 2020

@author: yaeger
"""
import numpy as np
from imblearn.metrics import geometric_mean_score
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score
from hyperopt import STATUS_OK
from sklearn.svm import SVC
import csv
from hyperopt.hp import lognormal, loguniform, uniform
from pathlib import Path
from hyperopt import Trials, tpe, fmin

def repeated_cross_val(estimator, x: np.ndarray, y: np.ndarray,
                       metric: callable = make_scorer(geometric_mean_score), 
                       repetitions: int = 5, cv_fold: int = 2,
                       **params):
    """Performs repeated cross-validation using the estimator, which must have
    a fit method.
    
    INPUTS:
        estimator: classifier with fit method
        x: x data
        y: labels
        metric: function to be used as score metric
        repetitions: number of times to repeat cross validation
        cv_fold: fold for cross validation
    RETURNS:
        mean metric score function on data        
    """
    scores = []
    for rep in range(repetitions):
        scores.extend(cross_val_score(estimator(**params),x,y,cv=cv_fold,scoring=metric,n_jobs =4))
    return np.mean(scores)

class BayesianOptimizer():
    """
    Wrapper class for use with hyperopt package.
    
    INPUTS:
        estimator: classifier model to be used
        x: numpy array of x training values
        y: numpy array of y training values
        metric: metric to be optimizied on validation set
        savepath: path or string to where results should be saved
        max_evals: how many iterations are allowed for finding best parameters
        repetitions: how many times cross-validation should be repeated and averaged
        cv_fold: what fold cross-validation should be performed
        variables: list of variables to optimize
        distributions: distributions to use for each variable (by order)
        arguments: arguments to each distribution-generating function as tuple
        variable_type: dictionary in which keys are variables and value is
            whether variable applies to estimator or sampler
        
    Builds a .csv file at the path specified by savepath which can be used to
    find the best model parameters.
    
    Also has a results attribute which can also be used to find the best model
    parameters.
    """
    def __init__(self: callable, estimator, x: np.ndarray, y: np.ndarray,
                 sampler: callable = None,
                 metric: callable = make_scorer(geometric_mean_score),
                 savepath: str = '/Users/yaeger/Documents/Modules/Porphyria/results/test',
                 max_evals: int = 50,
                 repetitions: int = 5, cv_fold: int = 2,
                 variables: list = ['C', 'gamma'],
                 distributions: list = ['loguniform','loguniform'],
                 arguments: list = [(0.1, 10),(1e-6,2)],
                 variable_type: dict = {'C':'estimator','gamma':'estimator'}):
        self.sampler = sampler
        self.x = x
        self.y = y
        self.estimator = estimator
        self.metric = metric
        self.max_evals = max_evals
        self.variable_type = variable_type
        
        if not isinstance(savepath,Path): savepath = Path(savepath)
        self.savepath = savepath
        
        with savepath.open('w') as fh:
            writer = csv.writer(fh)
            writer.writerow(['loss', 'params', 'iteration'])
            
        self.iteration = 0
        self.bayes_trials = Trials()
        self.repetitions = repetitions
        self.cv_fold = cv_fold
        
        # create domain space
        self.space = self.create_domain_space(variables,distributions,arguments)
        
        # if sampler doesn't have parameters, just sample once
        if self.sampler is not None:
            if 'sampler' not in self.variable_type.values():
                self.x, self.y = self.sampler(self.x, self.y)
            
        
    def objective(self, params):
        """Objective function for Hyperparameter optimization
        """
        self.iteration += 1
        
        if 'cost' in self.variable_type.values():
            cost = params['cost']
            del params['cost']
            # minority class has label of +1
            params['class_weight'] = {1: cost, -1: 1}
        
        if 'sampler' not in self.variable_type.values():
            # if no sampling parameters, no need to sample
            metric_result = repeated_cross_val(self.estimator, self.x, self.y, **params)
        else:
            # pull out sampler parameters
            sampler_params = {param: params[param] for param in params if \
                              self.variable_type[param] == 'sampler'}
            # sample
            x,y = self.sampler(self.x, self.y, **sampler_params)
            
            # pull out estimator params
            estimator_params = {param: params[param] for param in params if \
                              self.variable_type[param] == 'estimator'}
            
            metric_result = repeated_cross_val(self.estimator, x, y, **estimator_params)
        
        # make metric_result negative for optimization
        loss = 1 - metric_result
        
        # write results to csv file
        with self.savepath.open('a') as fh:
            writer = csv.writer(fh)
            writer.writerow([loss, params, self.iteration])
        
        return {'loss': loss, 'params': params, 'iteration': self.iteration,
                'status': STATUS_OK}
    
    def optimize_params(self):
        """ Wrapper method for fmin function in hyperopt package 
        """
        best = fmin(fn = self.objective, space = self.space, algo = tpe.suggest, 
            max_evals = self.max_evals, trials = self.bayes_trials, rstate = np.random.RandomState(50))
    
    @property
    def results(self):
        return self.bayes_trials.results
    
    @staticmethod
    def create_domain_space(variables: list = ['C', 'gamma'], 
                     distributions: list = ['loguniform','loguniform'], 
                     arguments: list = [(0.1, 10),(1e-6,10)]):
        """ Returns dictionary keyed by variable with the distribution
            
            INPUTS:
                variables: list of string variables
                
                distributions: list of string names of functions to generate 
                distributions
                
                arguments: list of tuples where each tuple contains arguments
                to distribution-generating function in order
            
            RETURNS:
                dictionary keyed by parameter type with specified distribution
                functions as keys
        """
        space = {}
        for i,variable in enumerate(variables):
            if distributions[i] == 'loguniform':
                (low, hi) = arguments[i]
                low, hi = np.log(low), np.log(hi)
                space[variable] = loguniform(variable,low,hi)
            elif distributions[i] == 'lognormal':
                mu, sigma = arguments[i]
                space[variable] = loguniform(variable,mu,sigma)
            elif distributions[i] == 'uniform':
                (low, high) = arguments[i]
                space[variable] = uniform(variable,low,high)
        return space
    
        
    
    
    
    
        
        