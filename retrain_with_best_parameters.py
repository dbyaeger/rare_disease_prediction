#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  7 12:53:03 2020

@author: yaeger
"""

from trainer import Trainer
from model_parameters.model_parameters import make_model_param_list


def run_retraining(params = {
                            'Mahalanobis_Distance_KNN': {'K':1201,
                                                         'k': 498,
                                                         'alpha': 0.005845451820124326}}):
    """wrapper function for TrainingMain. Creates list of parameterized models
    using the make_model_param_list function and then trains each model
    using the train_and_save_model method of the TrainingMain class.
    """
    model_parameter_list = make_model_param_list()
    for model_params in model_parameter_list:
        if model_params['model_name'] in list(params.keys()):
            model_name = model_params['model_name']
            model_trainer = Trainer(**model_params)
            model_trainer.train_and_save_model(params[model_name])

if __name__ == "__main__":
    run_retraining()