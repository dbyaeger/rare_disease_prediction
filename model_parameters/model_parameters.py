#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 13:41:35 2020

@author: yaeger

Set of functions and commands to create, run, and save each model.
"""

from sklearn.svm import SVC
from instance_methods.InstanceMethods import FaultDetectionKNN
from imblearn.metrics import geometric_mean_score
from sklearn.metrics import make_scorer, average_precision_score
from model_parameters.sampling_functions import (tomek_links, one_sided_selection, 
                                                random_undersample, smote,
                                                random_undersample_smote,
                                                random_oversample)

# parameters common to all models
common_params = {'path_to_data': '/Users/yaeger/Documents/Porphyria',
                 'save_training_data_path': '/Users/yaeger/Documents/Modules/Porphyria/results/training',
                 'save_model_path': '/Users/yaeger/Documents/Modules/Porphyria/models',
                 'metric': make_scorer(average_precision_score,needs_threshold=True),
                 'max_evals': 100,
                 'repetitions': 5,
                 'cv_fold': 2}

# params for SVC with no sampling strategy
svc = {'classifier': SVC, 'model_name': 'SVC', 'sampling_method': None,
              'preprocessing_method': None,
              'log_normalize': True, 
              'variables': ['C', 'gamma'],
              'distributions': ['uniform','loguniform'],
              'arguments': [(0, 100),(1e-3,3)], 
              'variable_type': {'C':'estimator','gamma':'estimator'}}

# params for SVC with TomekLinks sampling strategy
svc_tomek_links = {'classifier': SVC, 'model_name': 'SVC_TomekLinks',
              'preprocessing_method': None,
              'sampling_method': tomek_links,
              'log_normalize': True, 
              'variables': ['C', 'gamma'],
              'distributions': ['uniform','loguniform'],
              'arguments': [(0, 100),(1e-3,3)], 
              'variable_type': {'C':'estimator','gamma':'estimator'}}

# params for SVC with one-sided sampling strategy
svc_one_sided = {'classifier': SVC, 'model_name': 'SVC_One_Sided_Selection',
              'preprocessing_method': None,
              'sampling_method': one_sided_selection,
              'log_normalize': True, 
              'variables': ['C', 'gamma'],
              'distributions': ['uniform','loguniform'],
              'arguments': [(0, 100),(1e-3,3)], 
              'variable_type': {'C':'estimator','gamma':'estimator'}}

# params for simple random undersampling
svc_random_undersample = {'classifier': SVC, 'model_name': 'SVC_Random_Undersample', 
              'preprocessing_method': None,
              'sampling_method': random_undersample,
              'log_normalize': True, 
              'variables': ['sampling_strategy','C', 'gamma'],
              'distributions': ['uniform','uniform','loguniform'],
              'arguments': [(0,1),(0, 100),(1e-3,3)], 
              'variable_type': {'sampling_strategy': 'sampler',
                                'C':'estimator','gamma':'estimator'}}

# params for adjusting class weights
svc_cost =  {'classifier': SVC, 'model_name': 'SVC_Different_Costs',
             'preprocessing_method': None,
              'sampling_method': None,
              'log_normalize': True, 
              'variables': ['class_weight','C', 'gamma'],
              'distributions': ['uniform','uniform','loguniform'],
              'arguments': [(1,1e6),(0, 100),(1e-3,3)], 
              'variable_type': {'class_weight': 'estimator',
                                'C':'estimator','gamma':'estimator'}}

# params for SMOTE with no undersampling of majority class
svc_SMOTE = {'classifier': SVC, 'model_name': 'SVC_SMOTE', 
             'preprocessing_method': None,
              'sampling_method': smote,
              'log_normalize': True, 
              'variables': ['sampling_strategy','C', 'gamma'],
              'distributions': ['uniform','uniform','loguniform'],
              'arguments': [(0,1),(0, 100),(1e-3,3)], 
              'variable_type': {'sampling_strategy': 'sampler',
                                'C':'estimator','gamma':'estimator'}}

# params for SMOTE with different costs
svc_SMOTE_cost = {'classifier': SVC, 'model_name': 'SVC_SMOTE_Different_Costs', 
              'preprocessing_method': None,
              'sampling_method': smote,
              'log_normalize': True, 
              'variables': ['sampling_strategy','C', 'gamma','class_weight'],
              'distributions': ['uniform','uniform','loguniform','uniform'],
              'arguments': [(0,1),(0, 100),(1e-3,3), (0,1e6)], 
              'variable_type': {'sampling_strategy': 'sampler',
                                'C':'estimator','gamma':'estimator',
                                'class_weight': 'estimator'}}

# params for random_undersample followed by SMOTE
svc_random_undersample_smote = {'classifier': SVC, 
              'preprocessing_method': None,
              'model_name': 'SVC_Random_Undersample_SMOTE', 
              'sampling_method': random_undersample_smote,
              'log_normalize': True, 
              'variables': ['sampling_strategy_1','sampling_strategy_2',
                            'C', 'gamma'],
              'distributions': ['uniform','uniform','uniform','loguniform'],
              'arguments': [(0,1),(0,1),(0, 100),(1e-3,3)], 
              'variable_type': {'sampling_strategy_1': 'sampler',
                                'sampling_strategy_2': 'sampler',
                                'C':'estimator','gamma':'estimator'}}

# params for fault detection-KNN
fd_knn = {'classifier': FaultDetectionKNN,
          'preprocessing_method': None,
          'model_name': 'Fault_Detection_KNN', 
          'sampling_method': None,
          'log_normalize': True, 
          'variables': ['k','alpha'],
          'distributions': ['quniform','uniform'],
          'arguments': [(2,1000,1),(0,0.01)], 
          'variable_type': {'k': 'estimator', 'alpha': 'estimator'}}

# params for random undersample and cost
svc_random_undersample_cost = {'classifier': SVC, 
              'model_name': 'SVC_Random_Undersample_Different_Costs',
              'preprocessing_method': None,
              'sampling_method': random_undersample,
              'log_normalize': True, 
              'variables': ['class_weight','C', 'gamma', 'sampling_strategy'],
              'distributions': ['uniform','uniform','loguniform','uniform'],
              'arguments': [(1,1e6),(0, 100),(1e-3,3),(0,1)], 
              'variable_type': {'class_weight': 'estimator',
                                'C':'estimator','gamma':'estimator',
                                'sampling_strategy': 'sampler'}}

# params for simple random oversampling
svc_random_oversample = {'classifier': SVC, 'model_name': 'SVC_Random_Oversample', 
              'sampling_method': random_oversample,
              'preprocessing_method': None,
              'log_normalize': True, 
              'variables': ['sampling_strategy','C', 'gamma'],
              'distributions': ['uniform','uniform','loguniform'],
              'arguments': [(0,1),(0, 100),(1e-3,3)], 
              'variable_type': {'sampling_strategy': 'sampler',
                                'C':'estimator','gamma':'estimator'}}




def make_model_param_list(input_list: list = [#svc,svc_tomek_links,
                                              #svc_one_sided,
                                              #svc_random_undersample,
                                              #svc_random_undersample_cost,
                                              svc_cost,
                                              svc_SMOTE,
                                              svc_SMOTE_cost,
                                              svc_random_undersample_smote,
                                              fd_knn,
                                              svc_random_oversample],
                    common_params: dict = common_params):
    for model_param in input_list: model_param.update(common_params)
    return input_list

    
              
