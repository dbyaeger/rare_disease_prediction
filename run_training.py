#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 15:49:32 2020

@author: yaeger
"""
from trainer import Trainer
from model_parameters.model_parameters import make_model_param_list

def run_training():
    """wrapper function for TrainingMain. Creates list of parameterized models
    using the make_model_param_list function and then trains each model
    using the train_and_save_model method of the TrainingMain class.
    """
    model_parameter_list = make_model_param_list()
    print(model_parameter_list)
    for model_params in model_parameter_list:
        model_trainer = Trainer(**model_params)
        model_trainer.train_and_save_model()

if __name__ == "__main__":
    run_training()