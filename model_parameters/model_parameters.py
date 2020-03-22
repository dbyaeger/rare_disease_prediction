#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 13:41:35 2020

@author: yaeger

Set of functions and commands to create, run, and save each model.
"""

from sklearn.svm import SVC
from imblearn.metrics import geometric_mean_score, average_precision_score
from imblearn.under_sampling import TomekLinks
from sklearn.metrics import make_scorer
from sampling_functions import tomek_links, one_sided_selection, random_undersample

# parameters common to all models
common_params = {'path_to_data': '/Users/yaeger/Documents/Porphyria',
                 'save_training_data_path': '/Users/yaeger/Documents/Modules/Porphyria/results',
                 'save_model_path': '/Users/yaeger/Documents/Modules/Porphyria/models',
                 'metric': make_scorer(average_precision_score,needs_threshold=True),
                 'max_evals': 50,
                 'repetitions': 5,
                 'cv_fold': 2}

# params for SVC with no sampling strategy
svc = {'classifier': SVC, 'model_name': 'SVC', 'sampling_method': None,
              'log_normalize': True, 
              'variables': ['C', 'gamma'],
              'distributions': ['uniform','loguniform'],
              'arguments': [(0, 100),(1e-3,3)], 
              'variable_type': {'C':'estimator','gamma':'estimator'}}

# params for SVC with TomekLinks sampling strategy
svc_tomek_links = {'classifier': SVC, 'model_name': 'SVC_TomekLinks', 
              'sampling_method': tomek_links,
              'log_normalize': True, 
              'variables': ['C', 'gamma'],
              'distributions': ['uniform','loguniform'],
              'arguments': [(0, 100),(1e-3,3)], 
              'variable_type': {'C':'estimator','gamma':'estimator'}}

# params for SVC with one-sided sampling strategy
svc_one_sided = {'classifier': SVC, 'model_name': 'SVC_One_Sided_Selection', 
              'sampling_method': one_sided_selection,
              'log_normalize': True, 
              'variables': ['C', 'gamma'],
              'distributions': ['uniform','loguniform'],
              'arguments': [(0, 100),(1e-3,3)], 
              'variable_type': {'C':'estimator','gamma':'estimator'}}

# params for simple random undersampling
svc_random_undersample = {'classifier': SVC, 'model_name': 'SVC_Random_Undersample', 
              'sampling_method': random_undersample,
              'log_normalize': True, 
              'variables': ['sampling_strategy','C', 'gamma'],
              'distributions': ['uniform','uniform','loguniform'],
              'arguments': [(0,1),(0, 100),(1e-3,3)], 
              'variable_type': {'sampling_strategy': 'sampler',
                                'C':'estimator','gamma':'estimator'}}



def make_model_param_list(input_list: list = [svc,svc_tomek_links,svc_one_sided,
                                              svc_random_undersample],
                    common_params: dict = common_params):
    return list(map(lambda x, y=common_params: x.update(y),input_list))

    
              
