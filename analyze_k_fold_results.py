#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 09:50:09 2020

@author: yaeger
"""
import pandas as pd
import ast
import numpy as np
from pathlib import Path
from scipy.stats import wilcoxon
from scipy.stats import mannwhitneyu

def extract_and_rank_data(path_to_data: str, name_of_results: str = 'k_fold_results.csv', 
                 metric: str = 'test_fraction_correctly_ranked',
                 summary_df: bool = True):
    """Loads the results data and computes several statistics for each variable"""
    
    if not isinstance(path_to_data, Path): path_to_data = Path(path_to_data)
    
    df = pd.read_csv(path_to_data.joinpath(name_of_results))
    
    assert metric in df.columns, f'{metric} not available in .csv file!'
    
    # Collect model parameters
    results = {}
    for model in df['model_name'].values:
        results[model] = {}
        results[model]['values'] = ast.literal_eval(df[df['model_name'] == model][metric].values[0])
    
    # Get ranks
    for i in range(len(results[list(results.keys())[0]]['values'])):
        to_rank = []
        
        # Sort
        for model in df['model_name'].values:
            to_rank.append(results[model]['values'][i])
        to_rank.sort(reverse=True)
        
        for model in df['model_name'].values:
            if i == 0:
                results[model]['ranks'] = [to_rank.index(results[model]['values'][0]) + 1]
            else:
                results[model]['ranks'].append(to_rank.index(results[model]['values'][i]) + 1)
    
    if summary_df:
        summary_dict = {}
        summary_dict['model'] = [model for model in sorted(list(df['model_name'].values))]
        summary_dict[metric] = [f'{round(np.mean(results[model]["values"]),2)} +/- {round(np.std(results[model]["values"]),2)}' \
                    for model in sorted(list(df['model_name'].values))]
        summary_dict['rank'] = [f'{round(np.mean(results[model]["ranks"]),1)} +/- {round(np.std(results[model]["ranks"]),1)}' \
                    for model in sorted(list(df['model_name'].values))]
    
    with path_to_data.joinpath('10_fold_results.csv').open('w') as f:
        pd.DataFrame.from_dict(summary_dict).to_csv(f, index = False)
            
    return results

def analyze_data(results_dict: dict, test: callable = wilcoxon):
    """Takes in a dictionary of results keyed by model name, with 'values' as
    value (list of values for a metric) and reutrns a dataframe. Compares each
    model to the base SVC model."""
    stats_results = {'comparison': [], 'p_value': []}
    
    for model in list(results_dict.keys()):
        if model != 'SVC':
            stats_results['comparison'].append(f'SVC v. {model}')
            stats_results['p_value'].append(wilcoxon(results_dict['SVC']['values'],
                                            results_dict[model]['values'],
                                            alternative = 'less')[1])
            
    return stats_results
            
            
            
            
    
        
    
    
    