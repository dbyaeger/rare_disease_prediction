#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 09:50:43 2020

@author: yaeger
"""
from utils.dataset_helpers import dataset_preprocessing_and_partitioning, make_test_set
from utils.testing_helpers import proportion_correctly_ranked, convert_labels_to_integers
from sklearn.metrics import average_precision_score, roc_auc_score
import pandas as pd
from pathlib import Path
import re

class TestScores():
    def __init__(self, path_to_scores: str = '/Users/yaeger/Documents/Porphyria/selected_patients_signed_rank_csvs', 
                 path_to_data: str = '/Users/yaeger/Documents/Porphyria',
                 save_path: str = '/Users/yaeger/Documents/Modules/Porphyria/results/test_scores',
                 save_name: str = 'results_with_original_svm',
                 metrics: list = [average_precision_score, roc_auc_score]):
        
        self.path_to_scores = self.convert_to_path(path_to_scores, make_directory = False)
        self.path_to_data = self.convert_to_path(path_to_data, make_directory = False)
        self.save_path = self.convert_to_path(save_path, make_directory = True)
        self.save_path = self.save_path.joinpath(save_name)
        self.metrics = metrics
        
        self.data = self.filter_and_sort_by_zcode()
                
    def evaluate(self, save_results = True):
        """
        Evaluates the data on the list of metrics. 
           
            RETURNS: results_dict: results of evaluating the model in the format:
                    
                    {model_name: <model_name>, <metric>: float, ...}
           
        If save_results set to True, saves results as a .csv file at the
        location specified by save_path with the name specified by save_name.
        """
        results_dict = {'model_name': ['original_svm']}
        ranks = self.data['SIGNED_RANK_VALUE'].to_numpy()
        y = self.data['Category'].to_numpy()
        
        # Evaluate classifier on pairwise ranking task
        results_dict['fraction_correctly_ranked'] = \
                                        [proportion_correctly_ranked(ranks,y)]
        
         # Transform into pairwise ranking task
        y_true = convert_labels_to_integers(y)
        
        # Evaluate classifiers on each metric
        for metric_fnx in self.metrics:
            metric_name = re.findall(r'function (.*) at', str(metric_fnx))[0]
            results_dict[metric_name] = [metric_fnx(y_true,ranks)]
        
        if save_results:
            results_df = pd.DataFrame.from_dict(results_dict)
            results_df.to_csv(self.save_path, header=True, index=False)
        
        return results_dict
        
        
        
    def filter_and_sort_by_zcode(self):
        """Returns pandas DataFrame with ZCODE, Category, and SIGNED_RANK_VALUE
        """
        test_set = self.load_test_set(self.path_to_data)
        
        ranks = self.load_scores(self.path_to_scores)
        
        rank_with_categories = pd.merge(test_set,ranks,how = 'inner', on = 'ZCODE')
        
        return rank_with_categories
        
        
    @staticmethod
    def load_test_set(path_to_data: Path):
        """Loads test data using the specified path. Returns pandas Dataframe
        with ZCODE and Category as columns.
        """
        holdout_set, _ = dataset_preprocessing_and_partitioning(path_to_data)
        
        # Remove deceased and patients with previous meaningful mention of Porph
        test_set = make_test_set(holdout_set)
                
        return test_set[['ZCODE','Category']]
    
    @staticmethod
    def load_scores(path_to_scores: Path):
        """Loads score data using the specified path. Returns score data as a
        data frame with ZCODE and SIGNED_RANK_VALUE as columns"""
        
        with path_to_scores.joinpath('classify_svmlight_rbf_gamma_0.04_selected_alnylam_feature_modelers_only_non_aip_featuresv2_optimize.dump').open('r') as fh:
            ranks = pd.read_csv(fh, sep='\t')[['PMID', 'SIGNED_RANK_VALUE']]
        
        # Rename PMID to ZCODE to facilitate join
        ranks = ranks.rename(columns = {'PMID': 'ZCODE'})
        
        return ranks
        
    @staticmethod
    def convert_to_path(path: str, make_directory: bool = True):
        """Converts an input string to path and creates the directory if it
        does not already exist"""
        if not isinstance(path, Path): path = Path(path)
        
        if make_directory:
            if not path.is_dir():
                path.mkdir()
        
        return path
    
    @property
    def get_zcodes(self):
        """Returns zcodes common to test set and ranked set"""
        return self.data['ZCODE'].to_numpy()
        
    