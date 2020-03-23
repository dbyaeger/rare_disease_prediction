#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 09:27:48 2020

@author: yaeger
"""
from utils.training_helpers import BayesianOptimizer
from utils.dataset_helpers import dataset_preprocessing_and_partitioning
from utils.feature_helpers import log_and_normalize_features
from imblearn.metrics import geometric_mean_score
from sklearn.metrics import make_scorer
from pathlib import Path
import numpy as np
import joblib

class Trainer():
    """Wrapper class for loading and preprocessing data, training a model,
    and saving the trained model with the parameters optimizied by cross-validation
    
    INPUTS:
        classifier: classifier model to be used
        model_name: filename for saving model and output
        sampling_method: function to perform sampling
        log_normalize: if True, log-normalize variables before sampling or training
            classifier
        path_to_data: path to where data lives.
        save_training_data_path: path or string to where results should be saved.
            Results are the loss values on the CV set and corresponding
            classifier parameters when performing bayesian optimization of
            classifier.
        save_model_path: path or string to where models should be saved.
        metric: metric to be optimizied on validation set. Should be a callable
            function.
        max_evals: how many iterations are allowed for finding best parameters
        repetitions: how many times cross-validation should be repeated and averaged
        cv_fold: what fold cross-validation should be performed
        variables: list of variables to optimize
        distributions: distributions to use for each variable (by order)
        arguments: arguments to each distribution-generating function as tuple
        variable_type: dictionary in which keys are variables and value is
            whether variable applies to estimator or sampler
            
    OUTPUTS:
        Calling the train_and_save_model method will save a trained model
        with name <model_name> in <save_model_path> and a .csv file with
        the name <model_name> in <save_training_data_path> that shows the
        output of the Bayesian optimizer during training. The saved model is
        the model with the best parameters found by Bayesian optimization 
        trained on all of the training data.
                   
    """
    def __init__(self, classifier: callable, model_name: str,
                 sampling_method: callable = None, log_normalize: bool = True,
                path_to_data: str = '/Users/yaeger/Documents/Porphyria',
                save_training_data_path: str = '/Users/yaeger/Documents/Modules/Porphyria/results',
                save_model_path: str = '/Users/yaeger/Documents/Modules/Porphyria/models',
                metric: callable = make_scorer(geometric_mean_score),
                max_evals: int = 200,
                repetitions: int = 5, cv_fold: int = 2,
                variables: list = ['C', 'gamma'],
                distributions: list = ['uniform','loguniform'],
                arguments: list = [(0, 100),(1e-3,3)],
                variable_type: dict = {'C':'estimator','gamma':'estimator'}):
        
        self.classifier = classifier
        self.model_name = model_name
        self.save_training_data_path = self.convert_to_path(save_training_data_path)
        self.save_model_path = self.convert_to_path(save_model_path)
        self.metric = metric
        self.sampling_method = sampling_method
        self.max_evals = max_evals
        self.repetitions = repetitions
        self.cv_fold = cv_fold
        self.variables = variables
        self.distributions = distributions
        self.arguments = arguments
        self.variable_type = variable_type
        self.x, self.y = self.load_data(self.convert_to_path(path_to_data), 
                                        log_normalize)
    
    def train_and_save_model(self):
        """ Wrapper method to find the best parameters, retrain model using the
        best parameters, and save the model
        """
        # Set savepath to include model name
        savepath = self.save_training_data_path.joinpath(self.model_name)
        
        # Instantiate new bayesian optimizer
        bc = BayesianOptimizer(estimator = self.classifier, x = self.x, 
                               y = self.y, metric = self.metric,
                               sampler = self.sampling_method,
                               savepath = savepath,
                               max_evals = self.max_evals, cv_fold = self.cv_fold,
                               variables = self.variables,
                               distributions = self.distributions,
                               arguments = self.arguments,
                               variable_type = self.variable_type)
        
        # Train
        bc.optimize_params()
        
        # Find the best model and retrain using best parameters
        # bc.results is dictionary with loss and model parameters
        sorted_params_by_loss = sorted(bc.results, key = lambda x: x['loss'])
        
        # get best parameters
        best_params = sorted_params_by_loss[0]['params']
        
        # print best params
        for param in best_params:
            print(f'best value of {param}: {best_params[param]}')
        
        # Retrain model using best parameters
        classifer = self.classifier(**best_params)
        classifer.fit(self.x, self.y)
        print(f'Value of metric on entire train set for best model: \
              {geometric_mean_score(self.y,classifer.predict(self.x))}')
        
        # Save model
        savepath = self.save_model_path.joinpath(self.model_name)
        joblib.dump(classifer, savepath)
    
    @staticmethod
    def load_data(path_to_data: Path, log_normalize: bool = True, meta_data_columns: 
                                            list = ['ID','ZCODE','AIP_Diagnosis',
                                                    'Patient ID', 'Category',
                                                    'Porp_mention', 
                                                    'ABDOMINAL_PAIN_DX_NAME']):
        """Loads training data using the specified path. If log_normalize is
        set to True, log_normalizes the data. Removes metadata columns from
        data and returns training data and training labels as numpy arrays.
        """
        # Get training data
        holdout_set, training_data = dataset_preprocessing_and_partitioning(path_to_data)
        
        # Make y
        y = np.array(training_data['AIP_Diagnosis'])
        
        # Remove metadata columns from training_data
        columns_to_remove = []
        for column in meta_data_columns:
            if column in training_data.columns:
                columns_to_remove.append(column)
        
        training_data = training_data.drop(columns = columns_to_remove)
        
        if log_normalize:
            training_data = log_and_normalize_features(training_data)
        
        x = training_data.to_numpy()
        
        return x,y
 
    @staticmethod
    def convert_to_path(path: str):
        """Checks the type of path and converts to Path type if is not of type
        Path.
        """
        if not isinstance(path, Path):
            return Path(path)
        return path


    