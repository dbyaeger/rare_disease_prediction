#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 14:54:05 2020

@author: yaeger
"""
from pathlib import Path
import pandas as pd
import numpy as np
from utils.feature_helpers import log_and_normalize_features

def load_file(path: Path = Path('/Users/yaeger/Documents/Porphyria')):
    "Loads porphryia dataset and returns it as a dataframe"    
    
    with path.joinpath('aip_final_model_dump_data.tsv').open('r') as fh:
        data = pd.read_csv(fh, sep='\t')
    
    return data

def get_positive_IDs(path: Path = Path('/Users/yaeger/Documents/Porphyria')):
    "Loads patient IDs for patients with porphyria" 
    
    with path.joinpath('Porphyria_timeline.csv').open('r') as fh:
        data = pd.read_csv(fh)
    return data['Patient_ID'].values

def make_holdout_set(all_ID_df: pd.DataFrame, path: Path = Path('/Users/yaeger/Documents/Porphyria'),
                     n = 1000):
    """Creates a data frame with labels from the 100 manually reviewed
    porphyria cases which the model predicted to be close to the decision 
    boundary"""
    
    path = path.joinpath('csvs')
    
    # Get 100 Patient IDs manually reviewed with no mention of Porphyria
    with path.joinpath('Top50ScoredNoPorph.csv').open('r') as fh:
        no_porph_top50 = pd.read_csv(fh)[['Patient ID', 'Category']]
    
    with path.joinpath('Second50ScoredNoPorph.csv').open('r') as fh:
        no_porph_bottom50 = pd.read_csv(fh)[['Patient ID','Category']]
    
    # Combine top and bottom 50 into a dataframe
    no_porph_patients = [no_porph_top50,no_porph_bottom50]
    no_porph_patients = pd.concat(no_porph_patients)
    
    # Annotate with mention of Porphryia data
    no_porph_patients['Porph_mention'] = pd.Series(['No']*len(no_porph_patients),
                                            index = no_porph_patients.index)
    
    # Get 100 Patients manually reviewed with Porphyria mention
    with path.joinpath('Top50ScoredPorphNotesNoLab.csv').open('r') as fh:
        porph_top50 = pd.read_csv(fh)[['Patient ID', 'Category']]
    
    with path.joinpath('Second50PorphNotes80OthNoLab.csv').open('r') as fh:
        porph_bottom50 = pd.read_csv(fh)[['Patient ID','Category']]
    
    # Combine top and bottom 50 into a dataframe
    porph_patients = [porph_top50, porph_bottom50]
    porph_patients = pd.concat(porph_patients, ignore_index = True)
    
    # Annotate with mention of Porphryia data
    porph_patients['Porph_mention'] = pd.Series(['Yes']*len(porph_patients),
                                            index = porph_patients.index)
    # Combine Porph and No Porph dataframes
    annotated_patients = [porph_patients, no_porph_patients]
    annotated_patients = pd.concat(annotated_patients, ignore_index = True)
    
    
    # Rename Patient ID to ZCODE for merging purposes
    annotated_patients = annotated_patients.rename(columns = {'Patient ID': 'ZCODE'})
    
    # Get random IDs
    random_IDs = fetch_random_zscores(all_ID_df, n = n)
    
    # Select dataframe with random IDs
    random_IDs = all_ID_df[all_ID_df['ZCODE'].isin(random_IDs)]['ZCODE']
    
    # Append IDs to annotated dataframe
    annotated_patients = pd.merge(annotated_patients,random_IDs,how = 'outer', on = 'ZCODE')
    
    # Inner join with training data
    annotated_patients = pd.merge(annotated_patients,all_ID_df,how = 'inner', on = 'ZCODE')
    
    # Fix some spelling errors
    annotated_patients = annotated_patients.replace({'Category': 'Possilbe'},'Possible')
    annotated_patients = annotated_patients.replace({'Category': ['Unlikley', 'Unlikely ', 'Unlikely, unknown', 'Unlikly']}, 'Unlikely') 
    
    return annotated_patients

def get_top_200_scoring_IDs(path: Path = Path('/Users/yaeger/Documents/Porphyria')):
    "Returns the IDs from the Top50 and Second50 for with and without porph mention"
    path = path.joinpath('csvs')
    
    IDs = []
    
    for file_name in ['Top50ScoredNoPorph.csv', 
                      'Second50ScoredNoPorph.csv',
                      'Top50ScoredPorphNotesNoLab.csv', 
                      'Second50PorphNotes80OthNoLab.csv']:    
        with path.joinpath(file_name).open('r') as fh:
            data = pd.read_csv(fh)
        IDs.extend(data['Patient ID'].values)
    
    assert len(IDs) == 200, f'Expected 200 IDs but only {len(IDs)} found!'
    
    return IDs
            
        
def fetch_random_zscores(all_df: pd.DataFrame, n: int = 200, 
                         aip_positive_ids: np.array = get_positive_IDs(),
                         top_200_scoring_IDs: np.array = get_top_200_scoring_IDs()):
    """Returns n random ZSCORES that are not positive for AIP or in the top
        200 scoring patients.
    """
    eligible_IDs = list(set(all_df['ZCODE']) - set(aip_positive_ids) - set(top_200_scoring_IDs))
    
    return np.random.choice(eligible_IDs, size = n, replace = False)

def remove_holdout_set_from_training_set(training_df: pd.DataFrame, holdout_df: pd.DataFrame):
    "Removes the IDs in the holdout set from the training set"
    try:
        holdout_IDs = holdout_df['Patient ID'].values
    except:
        holdout_IDs = holdout_df['ZCODE'].values
    return training_df[~training_df.ZCODE.isin(holdout_IDs)]
    
def label_aip_diagnosis(all_df: pd.DataFrame, 
                       positive_patients: np.array = get_positive_IDs()):
    "Labels the training set"

    # Make the positive patients 
    aip_positive = pd.DataFrame({'ZCODE': positive_patients, 
                                 'AIP_Diagnosis': np.ones(len(positive_patients))})
    
    labeled_training_df = pd.merge(all_df, aip_positive, how = 'outer', on = 'ZCODE')
    
    return labeled_training_df.replace({'AIP_Diagnosis': np.nan}, -1)

def dataset_preprocessing_and_partitioning(path: Path = Path('/Users/yaeger/Documents/Porphyria')):
    "Wrapper function to load all data and pre-process"
    if not isinstance(path, Path): path = Path(path)
    
    # Load all data and features
    all_data = load_file(path)
    
    # Label data
    labeled_data = label_aip_diagnosis(all_data, positive_patients = get_positive_IDs(path))
    
    # Make holdout set
    holdout_set = make_holdout_set(labeled_data,path)
    
    # Remove holdout set from training data
    training_data = remove_holdout_set_from_training_set(labeled_data, holdout_set)
    
    return holdout_set, training_data

def make_test_set(holdout_set: pd.DataFrame) -> pd.DataFrame:
    """Takes the holdout set as input and creates the test set by:
        
        1) Deleting patients with 'Yes' mention of Porphyria, but for whom
        'Category' field is Possible. These patients had only an incidental
        mention of Porphyria in their notes.
        
        2) Deleting deceased patients.
        
        returns the transformed data frame.
    """
    
    no_porph_mention_living = holdout_set[(holdout_set['Porph_mention'] == 'No') \
                                          & (holdout_set['Category'] != 'Deceased')]
    
    yes_porph_mention_possible = holdout_set[(holdout_set['Porph_mention'] == 'Yes') \
                                          & (holdout_set['Category'] == 'Possible')]
    
    unlabeled = holdout_set[holdout_set['Porph_mention'].isnull()]
    unlabeled = unlabeled.replace({'Category': np.nan}, 'Unlikely')

    test_set = [no_porph_mention_living, yes_porph_mention_possible, unlabeled]
    return pd.concat(test_set,ignore_index=True)

def get_training_x_and_y(training_data: pd.DataFrame, log_normalize: bool = True,
                                 meta_data_columns: list = ['ID','ZCODE','AIP_Diagnosis',
                                'Patient ID', 'Porph_mention', 'ABDOMINAL_PAIN_DX_NAME']): 
    """Returns training data and labels as a numpy array. If log_normalize is
    set to True, also log-normalizes the data.
    """
    
    # Make y
    y = training_data['AIP_Diagnosis'].to_numpy()
    
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
    
    
    
    
    
    
    
    
    
    
    
    
    
    
        
    