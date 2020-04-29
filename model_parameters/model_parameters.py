#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 13:41:35 2020

@author: yaeger

Set of functions and commands to create, run, and save each model.
"""

from sklearn.svm import SVC
from instance_methods.InstanceMethods import FaultDetectionKNN, MahalanobisDistanceKNN
from shrink.shrink import SHRINK
from imblearn.metrics import geometric_mean_score
from sklearn.metrics import make_scorer, average_precision_score
from sklearn.decomposition import PCA, KernelPCA
from imblearn.under_sampling import (TomekLinks, OneSidedSelection, 
                                     RandomUnderSampler)
from imblearn.over_sampling import (SMOTE, RandomOverSampler, KMeansSMOTE,
                                    ADASYN)
from imblearn.ensemble import BalancedBaggingClassifier
RANDOM_STATE = 10
# parameters common to all models
common_params = {'path_to_data': '/Users/yaeger/Documents/Porphyria',
                 'save_training_data_path': '/Users/yaeger/Documents/Modules/Porphyria/results/training',
                 'save_model_path': '/Users/yaeger/Documents/Modules/Porphyria/models',
                 'metric': make_scorer(average_precision_score,needs_threshold=True),
                 'max_evals': 15,
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
              'sampling_method': TomekLinks(n_jobs=4),
              'log_normalize': True, 
              'variables': ['C', 'gamma'],
              'distributions': ['uniform','loguniform'],
              'arguments': [(0, 100),(1e-3,3)], 
              'variable_type': {'C':'estimator','gamma':'estimator'}}

# params for SVC with one-sided sampling strategy
svc_one_sided = {'classifier': SVC, 'model_name': 'SVC_One_Sided_Selection',
              'preprocessing_method': None,
              'sampling_method': OneSidedSelection(n_jobs=4, random_state = RANDOM_STATE),
              'log_normalize': True, 
              'variables': ['C', 'gamma'],
              'distributions': ['uniform','loguniform'],
              'arguments': [(0, 100),(1e-3,3)], 
              'variable_type': {'C':'estimator','gamma':'estimator'}}

# params for simple random undersampling
svc_random_undersample = {'classifier': SVC, 'model_name': 'SVC_Random_Undersample', 
              'preprocessing_method': None,
              'sampling_method': RandomUnderSampler(random_state = RANDOM_STATE),
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
              'sampling_method': SMOTE(n_jobs=4, random_state = RANDOM_STATE),
              'log_normalize': True, 
              'variables': ['sampling_strategy','k_neighbors','C', 'gamma'],
              'distributions': ['uniform','quniform','uniform','loguniform'],
              'arguments': [(0,1),(1,10,1),(0, 100),(1e-3,3)], 
              'variable_type': {'sampling_strategy': 'sampler',
                                'k_neighbors': 'sampler',
                                'C':'estimator','gamma':'estimator'}}

# params for SMOTE with different costs
svc_SMOTE_cost = {'classifier': SVC, 'model_name': 'SVC_SMOTE_Different_Costs', 
              'preprocessing_method': None,
              'sampling_method': SMOTE(n_jobs=4, random_state = RANDOM_STATE),
              'log_normalize': True, 
              'variables': ['sampling_strategy', 'k_neighbors',
                            'C', 'gamma','class_weight'],
              'distributions': ['uniform','quniform','uniform',
                                'loguniform','uniform'],
              'arguments': [(0,1),(1,10,1),(0, 100),(1,10), (0,1e6)], 
              'variable_type': {'sampling_strategy': 'sampler',
                                'k_neighbors': 'sampler',
                                'C':'estimator','gamma':'estimator',
                                'class_weight': 'estimator'}}

# params for fault detection-KNN
fd_knn = {'classifier': FaultDetectionKNN,
          'preprocessing_method': None,
          'model_name': 'Fault_Detection_KNN', 
          'sampling_method': None,
          'log_normalize': True, 
          'variables': ['k','alpha'],
          'distributions': ['quniform','uniform'],
          'arguments': [(2,500,1),(0,0.01)], 
          'variable_type': {'k': 'estimator', 'alpha': 'estimator'}}

# params for fault detection-KNN with linear pca
fd_knn_linear_pca = {'classifier': FaultDetectionKNN,
          'preprocessing_method': PCA,
          'model_name': 'Fault_Detection_KNN_linear_PCA', 
          'sampling_method': None,
          'log_normalize': False, 
          'variables': ['k','alpha', 'n_components'],
          'distributions': ['quniform','uniform', 'quniform'],
          'arguments': [(2,500,1),(0,0.01),(1,139,1)], 
          'variable_type': {'k': 'estimator', 'alpha': 'estimator',
                            'n_components': 'preprocessor'}}

# params for fault detection-KNN with radial PCA
fd_knn_radial_pca = {'classifier': FaultDetectionKNN,
          'preprocessing_method': KernelPCA(kernel="rbf", eigen_solver = "arpack"),
          'model_name': 'Fault_Detection_KNN_Radial_PCA', 
          'sampling_method': None,
          'log_normalize': False, 
          'variables': ['k','alpha', 'n_components','gamma'],
          'distributions': ['quniform','uniform', 'quniform','loguniform'],
          'arguments': [(2,500,1),(0,0.01),(1,139,1),(1e-6,300)], 
          'variable_type': {'k': 'estimator', 'alpha': 'estimator',
                            'n_components': 'preprocessor', 'gamma': 'preprocessor'}}

# params for adaptive Mahalanobis distance-KNN
mad_knn = {'classifier': MahalanobisDistanceKNN,
          'preprocessing_method': None,
          'model_name': 'Mahalanobis_Distance_KNN', 
          'sampling_method': None,
          'log_normalize': False, 
          'variables': ['K','k','alpha','precision_method'],
          'distributions': ['quniform','quniform','uniform','choice'],
          'arguments': [(10,2000,1),(2,500,1),(0,0.01),('MinCovDet', 'LedoitWolf')], 
          'variable_type': {'K': 'estimator', 'k': 'estimator', 
                            'alpha': 'estimator', 'precision_method': 'estimator'}}

# params for adaptive Mahalanobis distance-KNN
mad_knn_linear_pca = {'classifier': MahalanobisDistanceKNN,
          'preprocessing_method': PCA,
          'model_name': 'Mahalanobis_Distance_KNN_Linear_PCA', 
          'sampling_method': None,
          'log_normalize': False, 
          'variables': ['K','k','alpha','n_components','precision_method'],
          'distributions': ['quniform','quniform','uniform','quniform','choice'],
          'arguments': [(10,2000,1),(2,500,1),(0,0.01),(1,139,1),('MinCovDet', 'LedoitWolf')], 
          'variable_type': {'K': 'estimator', 'k': 'estimator', 
                            'alpha': 'estimator', 'n_components': 'preprocessor',
                            'precision_method': 'estimator'}}

# params for adaptive Mahalanobis distance-KNN
mad_knn_radial_pca = {'classifier': MahalanobisDistanceKNN,
          'preprocessing_method': KernelPCA(kernel="rbf", eigen_solver = "arpack"),
          'model_name': 'Mahalanobis_Distance_KNN_Radial_PCA', 
          'sampling_method': None,
          'log_normalize': False, 
          'variables': ['K','k','alpha','n_components','gamma','precision_method'],
          'distributions': ['quniform','quniform','uniform','quniform','loguniform',
                            'choice'],
          'arguments': [(10,2000,1),(2,500,1),(0,0.01),(1,139,1),(1e-6,300),
                        ('MinCovDet', 'LedoitWolf')], 
          'variable_type': {'K': 'estimator', 'k': 'estimator',
                            'alpha': 'estimator', 'n_components': 'preprocessor',
                            'gamma': 'preprocessor', 'precision_method': 'estimator'}}

# params for random undersample and cost
svc_random_undersample_cost = {'classifier': SVC, 
              'model_name': 'SVC_Random_Undersample_Different_Costs',
              'preprocessing_method': None,
              'sampling_method': RandomUnderSampler(random_state = RANDOM_STATE),
              'log_normalize': True, 
              'variables': ['class_weight','C', 'gamma', 
                            'sampling_strategy'],
              'distributions': ['uniform','uniform','loguniform','uniform'],
              'arguments': [(1,1e6),(0, 100),(1e-3,3),(0,1)], 
              'variable_type': {'class_weight': 'estimator',
                                'C':'estimator','gamma':'estimator',
                                'sampling_strategy': 'sampler'}}

# params for simple random oversampling
svc_random_oversample = {'classifier': SVC, 'model_name': 'SVC_Random_Oversample', 
              'sampling_method': RandomOverSampler(random_state = RANDOM_STATE), 
              'preprocessing_method': None,
              'log_normalize': True, 
              'variables': ['sampling_strategy','C', 'gamma'],
              'distributions': ['uniform','uniform','loguniform'],
              'arguments': [(0,1),(0, 100),(1e-3,3)], 
              'variable_type': {'sampling_strategy': 'sampler',
                                'C':'estimator','gamma':'estimator'}}

# params for kmeans smote
svc_KMeans_SMOTE = {'classifier': SVC, 'model_name': 'SVC_KMeans_SMOTE', 
             'preprocessing_method': None,
              'sampling_method': KMeansSMOTE(n_jobs=4, random_state = RANDOM_STATE),
              'log_normalize': True, 
              'variables': ['sampling_strategy','k_neighbors','cluster_balance_threshold',
                            'C', 'gamma'],
              'distributions': ['uniform','quniform','uniform','uniform','loguniform'],
              'arguments': [(0,1),(1,10,1),(0.1,3),(0, 100),(1e-3,3)], 
              'variable_type': {'sampling_strategy': 'sampler',
                                'k_neighbors': 'sampler',
                                'cluster_balance_threshold': 'sampler',
                                'C':'estimator','gamma':'estimator'}}


svc_KMeans_ADASYN = {'classifier': SVC, 'model_name': 'SVC_KMeans_ADASYN', 
             'preprocessing_method': None,
              'sampling_method': ADASYN(n_jobs=4, random_state = RANDOM_STATE),
              'log_normalize': True, 
              'variables': ['sampling_strategy','n_neighbors','C', 'gamma'],
              'distributions': ['uniform','quniform','uniform','loguniform'],
              'arguments': [(0,1),(1,10,1),(0, 100),(1e-3,10)], 
              'variable_type': {'sampling_strategy': 'sampler',
                                'n_neighbors': 'sampler',
                                'C':'estimator','gamma':'estimator'}}

shrink = {'classifier': SHRINK, 'model_name': 'SHRINK', 
             'preprocessing_method': None,
              'sampling_method': None,
              'log_normalize': False, 
              'variables': ['T','theta', 'metric_performance_threshold'],
              'distributions': ['quniform','uniform', 'uniform'],
              'arguments': [(1,61,1),(0,139),(0,0.45)], 
              'variable_type': {'T': 'estimator',
                                'theta': 'estimator',
                                'metric_performance_threshold': 'estimator'}}

voting_svm = {'classifier': BalancedBaggingClassifier(base_estimator = SVC(kernel='rbf'), 
                                                      n_jobs = 4, bootstrap = False,
                                                       random_state = RANDOM_STATE), 
              'model_name': 'BalancedVotingSVM', 
             'preprocessing_method': None,
              'sampling_method': None,
              'log_normalize': True, 
              'variables': ['sampling_strategy', 'n_estimators','C', 'gamma',
                            'replacement'],
              'distributions': ['uniform','quniform','uniform','loguniform',
                                'choice'],
              'arguments': [(0,1),(100,7000,1),(0, 100),(1e-3,1),(True,False)], 
              'variable_type': {'sampling_strategy': 'estimator',
                                'n_estimators': 'estimator',
                                'C': 'base_estimator',
                                'gamma': 'base_estimator',
                                'replacement': 'estimator'}}


def make_model_param_list(input_list: list = [#svc,svc_tomek_links,
                                              #svc_one_sided,
                                              #svc_random_undersample,
                                              #svc_random_undersample_cost,
                                              #svc_cost,
                                              #svc_SMOTE,
                                              #svc_SMOTE_cost,
                                              #shrink,
                                              fd_knn,
                                              fd_knn_linear_pca,
                                              voting_svm,
                                              fd_knn_radial_pca,
                                              mad_knn,
                                              mad_knn_linear_pca,
                                              mad_knn_radial_pca,
                                              svc_KMeans_SMOTE,
                                              svc_KMeans_ADASYN,
                                              svc_SMOTE_cost,
                                              svc_random_oversample],
                    common_params: dict = common_params):
    for model_param in input_list: model_param.update(common_params)
    return input_list

    
              
