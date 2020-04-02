#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 12:18:10 2020

@author: yaeger
"""
from utils.testing_helpers import proportion_correctly_ranked, convert_labels_to_integers
from utils.dataset_helpers import dataset_preprocessing_and_partitioning, make_test_set
from utils.feature_helpers import log_and_normalize_features
from pathlib import Path
from sklearn.metrics import make_scorer
import re
import joblib

class Tester():
    """Wrapper class for evaluating trained models on list of user-defined
    metrics. By default, all models are evaluated on the proportion of correctly
    ranked pairs.
    
        INPUTS:
            data_path: path to where data lives.
            metrics: list of metrics to be used in evaluating the model when
            framed as a two-class problem.
        OPTIONAL:
            model_path: path to where models live. Only necessary if only the names
            of models passed to evaluate_model function, rather than the full path.
        METHODS:
            evaluate_model takes the path to the model as a string or path, or
            the name of the saved model if the path is specified. Outputs a 
            dictionary with the names of the different metrics as keys and values
            of the score of the classifier on those metrics on the test set.
    """
    def __init__(self, path_to_data: str, metrics: list, model_path: str = None):
        
        if model_path is not None:
            self.model_path = self.convert_to_path(model_path, make_directory = False)
        
        self.path_to_data = self.convert_to_path(path_to_data, make_directory = False)
                
        if not isinstance(metrics, list):
            metrics = [metrics]
            
        self.metrics = list(map(make_scorer,metrics))
        
        self.x, self.y, self.zcodes = self.load_data(self.path_to_data)
    
    def evaluate_model(self, model_name: str):
        """ Loads the model, log transforms the data if required for the specific
        model, and evaluates the model over the list of metrics. and returns the
        results as a dictionary.:
        
            INPUT:
                model_name can either be a path or string specifying the path
                to the model, or it can be a string specifying the name of a model
                file living in the model_path directory.
            
            RETURNS:
                results_dict: results of evaluating the model in the format:
                    
                    {model_name: <model_name>, <metric>: float, ...}
        """
        # Load model and parameters
        if isinstance(model_name, Path):
            model, params = joblib.load(model_name)
        elif isinstance(model_name, str):
            try:
                model, params = joblib.load(model_name)
            except:
                model, params = joblib.load(self.model_path.joinpath(model_name))
        
        # Make results_dict to store results
        results_dict = {'model_name': params['model_name']}
        
        # Log-normalize data if necessary
        if params['log_normalize']:
            x = log_and_normalize_features(self.x)
        else:
            x = self.x
        
        # Get rankings and probabilities
        y_pred_rank = model.decision_function(x)
        
        # Evaluate classifier on pairwise ranking task
        results_dict['fraction_correctly_ranked'] = \
                                        proportion_correctly_ranked(y_pred_rank, 
                                                            self.y)
        # Transform into pairwise ranking task
        y_true = convert_labels_to_integers(self.y)
        
        # Evaluate classifiers on each metric
        for metric_fnx in self.metrics:
            metric_name = re.findall(r'\((.*)\)',str(metric_fnx))[0]
            results_dict[metric_name] = metric_fnx(model,x,y_true)
        
        return results_dict
                
    @staticmethod
    def load_data(path_to_data: Path, 
                  meta_data_columns: list = ['ID','ZCODE','AIP_Diagnosis',
                                             'Patient ID', 'Category',
                                             'Porph_mention', 
                                             'ABDOMINAL_PAIN_DX_NAME']):
        """Loads test data using the specified path. Removes metadata columns 
        from test data and returns test data, test labels, and z codes
        as numpy arrays.
        """
        holdout_set, _ = dataset_preprocessing_and_partitioning(path_to_data)
        
        # Remove deceased and patients with previous meaningful mention of Porph
        test_set = make_test_set(holdout_set)
        
        # Set categories as y labels
        y = test_set['Category'].to_numpy()
        
        # Get ZCODES
        zcodes = test_set['ZCODE'].to_numpy()
        
        # Remove metadata columns from training_data
        columns_to_remove = []
        for column in meta_data_columns:
            if column in test_set.columns:
                columns_to_remove.append(column)
        
        test_set = test_set.drop(columns = columns_to_remove)
        
        x = test_set.to_numpy()
        
        return x,y, zcodes
        
    @property
    def get_zcodes(self):
        """Returns the zcodes in the test set"""
        return self.zcodes
    
    @staticmethod
    def convert_to_path(path: str, make_directory: bool = True):
        """Converts an input string to path and creates the directory if it
        does not already exist"""
        if not isinstance(path, Path): path = Path(path)
        
        if make_directory:
            if not path.is_dir():
                path.mkdir()
        
        return path
