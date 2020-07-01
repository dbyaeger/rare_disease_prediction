#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 27 14:34:07 2020

@author: yaeger
"""
from train_and_test_k_fold import TrainTestKFold
from pathlib import Path
import numpy as np
from sklearn.metrics import average_precision_score
from imblearn.metrics import geometric_mean_score
import csv
import re
import pandas as pd
from imblearn.under_sampling import (TomekLinks, OneSidedSelection, 
                                     RandomUnderSampler)
from imblearn.over_sampling import (SMOTE, RandomOverSampler, KMeansSMOTE,
                                    ADASYN)
from samplers.samplers import ReturnMajority, ReturnMinority

def run_train_and_test_k_fold(path_to_data: str = '/Users/yaeger/Documents/Porphyria',
                              models_to_run: list = [
                                                     {'model_name': 'One_Class_SVM_Majority', 'sampler':  ReturnMajority()}],     
                              path_to_models: str = '/Users/yaeger/Documents/Modules/Porphyria/models',
                              save_directory: str = '/Users/yaeger/Documents/Modules/Porphyria/results/testing',
                              train_metrics: list = [average_precision_score, geometric_mean_score],
                              test_metrics: list = None,
                              results_file_name: str = 'k_fold_results.csv',
                              n_splits: int = 10):
    """ For each model in models_to_run, trains the model and saves the results 
    in a .csv file. By default, appends to the end of the .csv file with name
    results_file_name in save_directory.
    """
    if not isinstance(save_directory, Path): save_directory = Path(save_directory)
    
    # Create a csv file to store results
    metric_names = []
    
    if train_metrics:
       train_metricnames = list(map(lambda x: re.findall(r'function (.*) at', str(x))[0], train_metrics))
       train_metricnames = list(map(lambda x: f'training_{x}', train_metricnames))
       metric_names.extend(train_metricnames)
    
    if test_metrics:
        test_metricnames = list(map(lambda x: re.findall(r'function (.*) at', str(x))[0], test_metrics))
        test_metricnames = list(map(lambda x: f'test_{x}', test_metricnames))
        metric_names.extend(test_metricnames)
       
    metric_names.insert(0, 'test_fraction_correctly_ranked')
    metric_names.insert(0, 'model_name')
    
    if not save_directory.joinpath(results_file_name).exists():
        with save_directory.joinpath(results_file_name).open('w') as fh:
            writer = csv.writer(fh)
            writer.writerow(metric_names)

    # Instantiate TrainTestKFold object 
    ttk = TrainTestKFold(path_to_data = path_to_data,
                             train_metrics = train_metrics,
                             test_metrics = test_metrics,
                             model_path = path_to_models,
                             n_splits = n_splits)
    
    for item in models_to_run:
        print(f'Evaluating {item["model_name"]}')
        results_dict = ttk.evaluate_model(**item)
        
        with save_directory.joinpath(results_file_name).open('a') as fh:
                writer = csv.writer(fh)
                writer.writerow([results_dict[name] for name in metric_names])

if __name__  == "__main__":
    
        run_train_and_test_k_fold()
        