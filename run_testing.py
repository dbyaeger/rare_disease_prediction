#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 14:57:30 2020

@author: yaeger
"""
from tester import Tester
from pathlib import Path
from sklearn.metrics import (f1_score, average_precision_score, 
                             balanced_accuracy_score, roc_auc_score, 
                             precision_score, recall_score)
from imblearn.metrics import geometric_mean_score
import csv
import re


def run_testing(path_to_data: str = '/Users/yaeger/Documents/Porphyria/csvs',
                path_to_models: str = '/Users/yaeger/Documents/Modules/Porphyria/models',
                save_directory: str = '/Users/yaeger/Documents/Modules/Porphyria/results/training',
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
    
    # Instantiate Tester object
    tester = Tester(path_to_data = path_to_data, metrics = metrics)
    
    # Convert path_to_models to a Path and iterate over it to test all models
    # in directory
    if not isinstance(path_to_models, Path):
        path_to_models = Path(path_to_models)
        
    for model_path in path_to_models.iterdir():
        results_dict = tester.evaluate_model(model_path)
        with save_path.open('a') as fh:
            writer = csv.writer(fh)
            writer.writerow([results_dict[name] for name in metric_names])

    
    
    