#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 27 12:28:35 2020

@author: yaeger
"""
from utils.testing_helpers import proportion_correctly_ranked, convert_labels_to_integers
from utils.dataset_helpers import dataset_preprocessing_and_partitioning, make_test_set 
from utils.feature_helpers import log_and_normalize_features
from pathlib import Path
from sklearn.metrics import make_scorer
import re
import joblib
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold

class TrainTestKFold():
    """Wrapper class for evaluating models on list of user-defined
    metrics. For each model, trains the model using k-folds (i.e. trains
    on all but 1 of the folds) and evaluates both training and test set 
    performance. By default, all models are evaluated on the proportion of 
    correctly ranked pairs.
    
        INPUTS:
            
            data_path: path to where data lives.
            
            train_metrics: list of metrics to be used in evaluating the model 
            on the hold out training set when framed as a two-class problem.
            
            test_metrics: list of metrics to be used in evaluating the model 
            on the test set when framed as a two-class problem.
            
            n_splits: the number of k-fold splits to make
            
        OPTIONAL:
            
            model_path: path to where models live. Only necessary if only the names
            of models passed to evaluate_model function, rather than the full path.
            
        METHODS:
            
            evaluate_model takes the path to the model as a string or path, or
            the name of the saved model if the path is specified. Outputs a 
            dictionary with the names of the different metrics as keys and values
            of the score of the classifier on those metrics on the test set.
            
    """
    def __init__(self, path_to_data: str, train_metrics: list, test_metrics: list,
                 model_path: str = None, n_splits: int = 10, 
                 parameters_with_integer_values: list = ['n_estimators','K','k',
                                                         'n_components', 'k_neighbors']):
        
        if model_path is not None:
            self.model_path = self.convert_to_path(model_path, make_directory = False)
        
        self.path_to_data = self.convert_to_path(path_to_data, make_directory = False)
                
        if not isinstance(train_metrics, list) and train_metrics is not None:
            train_metrics = [train_metrics]
            
        if not isinstance(test_metrics, list) and test_metrics is not None:
            test_metrics = [test_metrics]
        
        if test_metrics is not None:
            self.test_metrics = list(map(make_scorer,test_metrics))
        else:
            self.test_metrics = None
        
        if train_metrics is not None:
            self.train_metrics = list(map(make_scorer,train_metrics))
        else:
            self.train_metrics = None
        
        self.n_splits = n_splits
        
        self.parameters_with_integer_values = parameters_with_integer_values
        
        self.test_x, self.test_y, self.zcodes = self.load_test_data(self.path_to_data)
        
        self.x, self.y = self.load_train_data(self.convert_to_path(path_to_data,
                                        make_directory = False))
    
    def evaluate_model(self, model_name: str, sampler: callable = None):
        """ Loads the model, log transforms the data if required for the specific
        model, and evaluates the model over the list of metrics. and returns the
        results as a dictionary.:
        
            INPUT:
                model_name can either be a path or string specifying the path
                to the model, or it can be a string specifying the name of a model
                file living in the model_path directory.
                
                sampler: the sampler to be used when resampling the training data.
            
            RETURNS:
                results_dict: results of evaluating the model in the format:
                    
                    {model_name: <model_name>, <metric>: float, ...}
                
                Also returns the parameters of the model in the format:
                    
                    {<parameter>: <value>, <parameter>: <value>, etc.}
        """
        # Load model and parameters
        if isinstance(model_name, Path):
            loaded = joblib.load(model_name)
        elif isinstance(model_name, str):
            try:
                loaded = joblib.load(model_name)
            except:
                loaded = joblib.load(self.model_path.joinpath(model_name))
                
        # Extract model and parameters
        model = loaded['model']
        params = self.integerize_params(loaded['training_parameters'])
        
        # Make results_dict to store results
        results_dict = {'model_name': params['model_name']}
        results_dict['test_fraction_correctly_ranked'] = []
        
        if self.train_metrics is not None:
            for metric_fnx in self.train_metrics:
                metric_name = re.findall(r'\((.*)\)',str(metric_fnx))[0]
                results_dict[f'training_{metric_name}'] = []
        
        if self.test_metrics is not None:
            for metric_fnx in self.test_metrics:
                metric_name = re.findall(r'\((.*)\)',str(metric_fnx))[0]
                results_dict[f'test_{metric_name}'] = []
        
        # Log-normalize data if necessary
        if params['log_normalize']:
            x = log_and_normalize_features(self.x)
            test_x = log_and_normalize_features(self.test_x)
        else:
            x = self.x
            test_x = self.test_x
        
        # Make into pipeline if preprocessing method
        if 'preprocessing_method' in params:
                pipeline_params = {}
                
                # Collect preprocessing params
                steps = [('preprocessor', params['preprocessing_method'])]
                preprocessing_params = {f'preprocessor__{param}': params[param] \
                              for param in params['variable_type'] if \
                              params['variable_type'][param] == 'preprocessor'}
                
                pipeline_params.update(preprocessing_params)
                
                # Collect estimator params
                steps.append(('estimator', model))
                clf_params = model.get_params()
                estimator_params = {f'estimator__{param}': clf_params[param] \
                                    for param in clf_params}
                pipeline_params.update(estimator_params)
                
                # Make pipeline
                model = Pipeline(steps)
                model.set_params(**pipeline_params)
        
        # Collect sampler params - DON'T PUT IN PIPEPLINE
        sampler_params = {param: params[param] for param in 
                          params['variable_type'] if \
                          params['variable_type'][param] == 'sampler'}
        
        # Make splitter
        skfd = StratifiedKFold(n_splits=self.n_splits, random_state=10, shuffle=True)
        
        for train_index, test_index in skfd.split(x, self.y):
            # train_test is "test" portion from the training set
            X_train, X_train_test = x[train_index], x[test_index]
            y_train, y_train_test = self.y[train_index], self.y[test_index]
            
            # resample if neccessary
            if sampler is not None:
                if sampler_params:
                    sampler.set_params(**sampler_params)
                
                X_train, y_train = sampler.fit_resample(X_train, y_train)
            
            # train model/pipeline
            model.fit(X_train,y_train)
            
            # Evaluate classifiers on each training metric
            for metric_fnx in self.train_metrics:
                metric_name = re.findall(r'\((.*)\)',str(metric_fnx))[0]
                results_dict[f'training_{metric_name}'].append(metric_fnx(model,X_train_test,y_train_test))
            
            # Evaluate classifiers on each test metric
            y_pred_rank = model.decision_function(test_x)
            
            # Evaluate classifier on pairwise ranking task
            results_dict['test_fraction_correctly_ranked'].append(proportion_correctly_ranked(y_pred_rank, 
                                                            self.test_y))
            if self.test_metrics is not None:
                
                # Transform into pairwise ranking task
                y_true = convert_labels_to_integers(self.test_y)
        
                # Evaluate classifiers on each metric
                for metric_fnx in self.test_metrics:
                    metric_name = re.findall(r'\((.*)\)',str(metric_fnx))[0]
                    results_dict[f'test_{metric_name}'].append(metric_fnx(model,test_x,y_true))
        
        return results_dict
    
    @staticmethod
    def load_train_data(path_to_data: Path, meta_data_columns: 
                                            list = ['ID','ZCODE','AIP_Diagnosis',
                                                    'Patient ID', 'Category',
                                                    'Porph_mention', 
                                                    'ABDOMINAL_PAIN_DX_NAME']):
        """Loads training data using the specified path. If log_normalize is
        set to True, log_normalizes the data. Removes metadata columns from
        data and returns training data and training labels as numpy arrays.
        """
        # Get training data
        _, training_data = dataset_preprocessing_and_partitioning(path_to_data)
        
        # Make y
        y = training_data['AIP_Diagnosis'].to_numpy()
        
        # Remove metadata columns from training_data
        columns_to_remove = []
        for column in meta_data_columns:
            if column in training_data.columns:
                columns_to_remove.append(column)
        
        training_data = training_data.drop(columns = columns_to_remove)
        
        x = training_data.to_numpy()
        
        return x,y
                
    @staticmethod
    def load_test_data(path_to_data: Path, 
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
    
    def integerize_params(self, params):
        new_params = {}
        for param in params:
            if param in self.parameters_with_integer_values:
                new_params[param] = int(params[param])
            else:
                new_params[param] = params[param]
        return new_params
            
    
    @staticmethod
    def convert_to_path(path: str, make_directory: bool = True):
        """Converts an input string to path and creates the directory if it
        does not already exist"""
        if not isinstance(path, Path): path = Path(path)
        
        if make_directory:
            if not path.is_dir():
                path.mkdir()
        
        return path
