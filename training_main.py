#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 09:27:48 2020

@author: yaeger
"""
from training_helpers import BayesianOptimizer
from dataset_helpers import dataset_preprocessing_and_partitioning
from feature_helpers import log_and_normalize_features
from imblearn.metrics import geometric_mean_score
from sklearn.metrics import make_scorer
from pathlib import Path
from sklearn.svm import SVC
import numpy as np
import pandas as pd
import joblib

class TrainingMain():
    """Wrapper class for loading and preprocessing data, training a model,
    and saving the trained model with the parameters optimizied by cross-validation
    """
    def __init__(self, classifier = SVC, model_name: str = 'SVC',
                 sampling_strategy: str = None, log_normalize: bool = True,
                path_to_data: str = '/Users/yaeger/Documents/Porphyria',
                save_training_data_path: str = '/Users/yaeger/Documents/Modules/Porphyria/results',
                save_model_path: str = '/Users/yaeger/Documents/Modules/Porphyria/models',
                metric: callable = make_scorer(geometric_mean_score),
                max_evals: int = 20,
                repetitions: int = 5, cv_fold: int = 2,
                variables: list = ['C', 'gamma'],
                distributions: list = ['loguniform','loguniform'],
                arguments: list = [(0.1, 100),(1e-3,1)]):
        
        self.classifier = classifier
        self.model_name = model_name
        self.sampling_strategy = sampling_strategy
        self.save_training_data_path = self.convert_to_path(save_training_data_path)
        self.save_model_path = self.convert_to_path(save_model_path)
        self.metric = metric
        self.max_evals = max_evals
        self.repetitions = repetitions
        self.cv_fold = cv_fold
        self.variables = variables
        self.distributions = distributions
        self.arguments = arguments
        self.x, self.y = self.load_data(self.convert_to_path(path_to_data), 
                                        log_normalize)
        if sampling_strategy is not None:
            self.x, self.y = self.sample()
    
    
    def sample(self):
        """Wrapper method for different sampling strategies
        """
        pass
    
    def train_and_save_model(self):
        """ Wrapper method to find the best parameters, retrain model using the
        best parameters, and save the model
        """
        # Set savepath to include model name
        savepath = self.save_training_data_path.joinpath(self.model_name)
        
        # Instantiate new bayesian optimizer
        bc = BayesianOptimizer(estimator = self.classifier, x = self.x, 
                               y = self.y, metric = self.metric, 
                               savepath = savepath,
                               max_evals = self.max_evals, cv_fold = self.cv_fold,
                               variables = self.variables,
                               distributions = self.distributions,
                               arguments = self.arguments)
        
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
        
#        y_pos = np.nonzero(y == 1)[0]
#        y_neg = np.nonzero(y == -1)[0][:2000]
#        idx = np.concatenate((y_pos,y_neg))
#        x = x[idx, :]
#        y = y[idx]
        
        return x,y
 
    @staticmethod
    def convert_to_path(path: str):
        """Checks the type of path and converts to Path type if is not of type
        Path.
        """
        if not isinstance(path, Path):
            return Path(path)
        return path

if __name__ == "__main__":
    tm = TrainingMain()
    tm.train_and_save_model()
    