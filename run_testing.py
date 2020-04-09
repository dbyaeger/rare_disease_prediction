#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 14:57:30 2020

@author: yaeger
"""
from tester import Tester
from pathlib import Path
import numpy as np
from sklearn.metrics import (f1_score, average_precision_score, 
                             balanced_accuracy_score, roc_auc_score, 
                             precision_score, recall_score)
from imblearn.metrics import geometric_mean_score
import csv
import re
import pandas as pd

def update_param_dict(new_dict: dict, master_dict: dict):
    """ Appends the entries of the new_dict as values to lists with the same keys
    in master_dict. If a new key is given in new_dict not appearing in master_dict,
    a new list will be created in master_dict with numpy nan's in the previous
    spots and the updated values in the last spot. Returns master_dict.
    """
    if not master_dict:
        master_dict = {key: [new_dict[key]] for key in new_dict}
    else:
        for key in new_dict:
            if key in master_dict:
                try:
                    master_dict[key].append(new_dict[key])
                except:
                    master_dict[key].extend(new_dict[key])
            else:
                max_len = max([len(master_dict[key]) for key in master_dict])
                master_dict[key] = [np.nan]*max_len
                try:
                    master_dict[key].append(new_dict[key])
                except:
                    master_dict[key].extend(new_dict[key])
    # Ensure that np.nans are added to all entries to keep length the same
    max_len = max([len(master_dict[key]) for key in master_dict])
    for key in master_dict:
        len_mismatch = max_len - len(master_dict[key])
        if len_mismatch > 0:
            master_dict[key].extend([np.nan]*len_mismatch)
    return master_dict
            

def run_testing(path_to_data: str = '/Users/yaeger/Documents/Porphyria',
                path_to_models: str = '/Users/yaeger/Documents/Modules/Porphyria/models',
                save_directory: str = '/Users/yaeger/Documents/Modules/Porphyria/results/testing',
                metrics: list = [geometric_mean_score, f1_score, average_precision_score,
                                 roc_auc_score, recall_score, precision_score,
                                 balanced_accuracy_score],
                results_file_name: str = 'test_set_results'):
    """Wrapper method for running the tester on all of the models in a directory
    for an input list of evaluation metrics. Writes a .csv to a file named
    according to the results_file_name argument with columns specifying the name
    of each model and each metric. Each row contains the results for each metric
    for a different model.
    
    INPUTS:
        path_to_data: where the data lives.
        path_to_models: where the models live.
        save_directory: the directory where the results should be saved. If this
            directory does not exist, it will be created.
        metrics: a list of metrics accepting y_true and y_pred (or y_score). These
            metrics should NOT accept a classifier as an argument.
        results_file_name: the name to use to save the results file.
    
    RETURNS:
        None
    
    """
    
    # Make a directory to store results
    if not isinstance(save_directory, Path):
        save_directory = Path(save_directory)
    if not save_directory.is_dir():
        save_directory.mkdir()
    
    save_path = save_directory.joinpath(results_file_name)
    
    # Create a csv file to store results
    metric_names = list(map(lambda x: re.findall(r'function (.*) at', str(x))[0], metrics))
    metric_names.insert(0, 'fraction_correctly_ranked')
    metric_names.insert(0, 'model_name')
    
    with save_path.open('w') as fh:
        writer = csv.writer(fh)
        writer.writerow(metric_names)
    
    # Create a dictionary to store parameter values
    param_dict = {}
    
    # Instantiate Tester object
    tester = Tester(path_to_data = path_to_data, metrics = metrics)
    
    # Convert path_to_models to a Path and iterate over it to test all models
    # in directory
    if not isinstance(path_to_models, Path):
        path_to_models = Path(path_to_models)
        
    for model_path in path_to_models.iterdir():
        if model_path.name == '.DS_Store':
            continue
        results_dict, parameters = tester.evaluate_model(model_path)
        param_dict = update_param_dict(parameters, param_dict)
        with save_path.open('a') as fh:
            writer = csv.writer(fh)
            writer.writerow([results_dict[name] for name in metric_names])
        
    # Save paramters
    param_directory = save_directory.joinpath(results_file_name + '_model_parameters')
    pd.DataFrame.from_dict(param_dict).to_csv(param_directory, index = False)

if __name__ == "__main__":
    run_testing()

    
    
    