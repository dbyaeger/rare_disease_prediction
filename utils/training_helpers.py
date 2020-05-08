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
from hyperopt.hp import lognormal, loguniform, uniform, quniform, choice
from pathlib import Path
from hyperopt import Trials, tpe, fmin
from imblearn.pipeline import Pipeline

def repeated_cross_val(estimator: callable, x: np.ndarray, y: np.ndarray,
                       params: dict, metric: callable = make_scorer(geometric_mean_score), 
                       repetitions: int = 5, cv_fold: int = 2):
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
        print(estimator)
        estimator.set_params(**params)
        scores.extend(cross_val_score(estimator,x,y,cv=cv_fold,scoring=metric,n_jobs =4))
    return np.mean(scores)

class BayesianOptimizer():
    """
    Wrapper class for use with hyperopt package.
    
    INPUTS:
        estimator: classifier model to be used
        sampler: sampling method to be used
        preprocessor: preprocessing method to be used
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
        parameters_with_integer_values: list of variable names that should be
        set to integer values.
        
    Builds a .csv file at the path specified by savepath which can be used to
    find the best model parameters.
    
    Also has a results attribute which can also be used to find the best model
    parameters.
    """
    def __init__(self: callable, estimator, x: np.ndarray, y: np.ndarray,
                 sampler: callable = None,
                 preprocessor: callable = None,
                 metric: callable = make_scorer(geometric_mean_score),
                 savepath: str = '/Users/yaeger/Documents/Modules/Porphyria/results/test',
                 max_evals: int = 50,
                 repetitions: int = 5, cv_fold: int = 2,
                 variables: list = ['C', 'gamma'],
                 distributions: list = ['loguniform','loguniform'],
                 arguments: list = [(0.1, 10),(1e-6,2)],
                 variable_type: dict = {'C':'estimator','gamma':'estimator'},
                 parameters_with_integer_values: list = ['n_estimators','K','k',
                                                         'n_components']):
        self.sampler = sampler
        self.preprocessor = preprocessor
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
        self.parameters_with_integer_values = parameters_with_integer_values
        
        # create domain space
        self.space = self.create_domain_space(variables,distributions,arguments)
           
    def objective(self, params):
        """Objective function for Hyperparameter optimization
        """
        self.iteration += 1
        
        preprocessing_params, sampler_params, estimator_params = self._sort_params(params)
        
        if not sampler_params and not preprocessing_params:
            # if no sampling parameters, no need to sample
            metric_result = repeated_cross_val(self.estimator, self.x, self.y, estimator_params)
        else:
            pipeline_classifier, pipeline_parameters = self.make_pipeline(preprocessing_params, sampler_params, estimator_params)
            metric_result = repeated_cross_val(pipeline_classifier, self.x, self.y, pipeline_parameters)
        
        # make metric_result negative for optimization
        loss = 1 - metric_result
        
        # write results to csv file
        with self.savepath.open('a') as fh:
            writer = csv.writer(fh)
            writer.writerow([loss, params, self.iteration])
        
        return {'loss': loss, 'params': params, 'iteration': self.iteration,
                'status': STATUS_OK}
    
    def _sort_params(self, params, dictionize_cost: bool = True,
                     majority_class: int = -1, minority_class: int = 1):
        """Separates parameters into parameters for sampling method and those
        for estimator. Also handles cost parameter and turns parameters named 
        in the attribute parameters_with_integer_values into integer values.
        Returns preprocessing_params, sampler_params, and estimator_params. 
        """
        # convert parameters in parameters_with_integer_values into integers
        for param in params:
            if param in self.parameters_with_integer_values:
                if not isinstance(params[param], int): 
                    params[param] = int(params[param])
        
        if dictionize_cost:
            if 'class_weight' in params:
                # minority class has label of +1. Class weights must be in the form of a dictionary
                params['class_weight'] = {minority_class: params['class_weight'], 
                                          majority_class: 1}
        
        preprocessing_params = {param: params[param] for param in params if \
                              self.variable_type[param] == 'preprocessor'}
        
        sampler_params = {param: params[param] for param in params if \
                              self.variable_type[param] == 'sampler'}
        
        estimator_params = {param: params[param] for param in params if \
                              self.variable_type[param] == 'estimator'}
        
        for param in estimator_params:
            if param in ['n_estimators']:
                estimator_params['n_estimators'] = int(estimator_params['n_estimators'])
        
        base_estimator_params = {f'base_estimator__{param}': params[param] for param in params if \
                              self.variable_type[param] == 'base_estimator'}
        
        if base_estimator_params:
            estimator_params.update(base_estimator_params)
        
        return preprocessing_params, sampler_params, estimator_params
    
    def make_pipeline(self, preprocessing_params, sampler_params, estimator_params):
        """Assembles a pipeline allowing the steps to be cross-validated 
        together. Takes in parameters for preprocessing, sampler, and
        estimator and returns a pipeline constructed from the preprocessing,
        sampler, and estimator parameters and the parameters."""
        steps = []
        pipeline_params = {}
        
        if sampler_params:
            steps.append(('sampler', self.sampler))
            sampler_pipeline_params = {f'sampler__{key}':sampler_params[key] \
                                       for key in sampler_params}
            pipeline_params.update(sampler_pipeline_params)
        
        if preprocessing_params:
            steps.append(('preprocessor', self.preprocessor))
            pipeline_preprocessing_params = {f'preprocessor__{key}':preprocessing_params[key] \
                                             for key in preprocessing_params}
            pipeline_params.update(pipeline_preprocessing_params)
        
        steps.append(('estimator', self.estimator))
        pipeline_estimator_params = {f'estimator__{key}':estimator_params[key] \
                                     for key in estimator_params}
        pipeline_params.update(pipeline_estimator_params)
        
        return Pipeline(steps), pipeline_params
        
    def train_and_return_model(self, params: dict, print_training_metric: bool = True):
        """Method to train model on full dataset with selected parameters,
        evaluate metric on training set, and return the model. If 
        print_training_metric set to True, will also print the performance of
        the model on the training set.
        """
        x,y = self.x, self.y
        preprocessing_params, sampler_params, estimator_params = self._sort_params(params, dictionize_cost = False)
        
        #classifier = self.estimator
        self.estimator.set_params(**estimator_params)
        
        if preprocessing_params:
            self.preprocessor.set_params(**preprocessing_params)
            try:
                x = self.preprocessor.fit_transform(x,y)
            except:
                x = self.preprocessor.fit_transform(x)
        
        if not sampler_params:
            # if no sampling parameters, no need to sample
            self.estimator.fit(x,y)
        else:
            self.sampler.set_params(**sampler_params)
            x,y = self.fit_resample(x, y)            
            self.estimator.fit(x,y)
        
        # Evaluate metric on train set
        if print_training_metric:
            
            if preprocessing_params:
                self.preprocessor.set_params(**preprocessing_params)
                try:
                    x = self.preprocessor.fit_transform(self.x,self.y)
                except:
                    x = self.preprocessor.fit_transform(self.x)
                    
            print(f'Value of metric on entire train set for best model: \
                  {self.metric(self.estimator,x,y)}')
            
        return self.estimator
           
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
            elif distributions[i] == 'quniform':
                (low, high, q) = arguments[i]
                space[variable] = quniform(variable,low,high,q)
            elif distributions[i] == 'choice':
                space[variable] = choice(variable, arguments[i])

        return space
    
        
    
    
    
    
        
        