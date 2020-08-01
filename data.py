# -*- coding: utf-8 -*-

"""A module for loading and preparing penguin data for machine learning."""

# Imports ---------------------------------------------------------------------

import datetime
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler

# Constants -------------------------------------------------------------------

RANDOM_STATE = 1999
DATA_FILE = 'data/palmer-penguins.csv'
DEFAULT_TARGET = 'Adelie'

# Load data -------------------------------------------------------------------

def load_data(
    filename=DATA_FILE, 
    add_target=True, 
    target_species=DEFAULT_TARGET):

    """
    Loads the Palmer penguins dataset, adds a target variable for the given 
    species, and removes the two cases with missing bill features.
    """
    
    df = pd.read_csv(DATA_FILE)

    # Add a target variable for the given species
    if add_target:
        df = add_target_species(df, target_species)

    # Remove the records with missing bill features
    df = df.query('bill_length_mm.notna() & bill_depth_mm.notna()')
    
    # Reset the index
    df = df.reset_index(drop=True)

    return df


def add_target_species(df, target):

    """
    Adds a binary target variable for the given species.
    """

    # Create the target variable
    df['target'] = df['species'].apply(lambda s: 1 if s == target else 0)

    return df

# Train and test sets ---------------------------------------------------------

def get_train_test_split(df):

    """
    Use a stratified shuffle split to get test and train datasets with
    an equal proportion of species in each set.
    """

    ssp = StratifiedShuffleSplit(
        n_splits=1, 
        test_size=0.3, 
        random_state = RANDOM_STATE)

    for train_ix, test_ix in ssp.split(df, df['species']):
        train = df.loc[train_ix]
        test = df.loc[test_ix]

    return train, test


def load_train_set(
    filename=DATA_FILE, 
    add_target=True, 
    target_species=DEFAULT_TARGET):

    """Load the train set"""
    df = load_data(filename, add_target, target_species)
    train, test = get_train_test_split(df)
    return train


def load_test_set(
    filename=DATA_FILE, 
    add_target=True, 
    target_species=DEFAULT_TARGET):

    """Load the test set"""
    df = load_data(filename, add_target, target_species)
    train, test = get_train_test_split(df)
    return test

# Prepare data for machine learning -------------------------------------------

def get_pipeline(
    df_X, 
    add_polynomials=False, 
    polynomial_degree=3):

    """
    Create and fit the pipeline for preparing the bill features. 

    This uses the pattern of combining separate transformers for differet sets 
    of features into a single pipeline with a ColumnTransformer, even though 
    only two numerical features are used which require the same preparation 
    steps. Similarly, the number transformer is set up to impute missing values 
    with the median of the feature in question, although in this particular 
    case rows with missing bill features are removed when the data is loaded.
    """

    # Create the number transforer
    number_features = [
        'bill_length_mm',
        'bill_depth_mm',
    ]

    if add_polynomials:
        number_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('poly_features', PolynomialFeatures(degree=polynomial_degree)),
            ('scaler', StandardScaler())
        ])
    else:
        number_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

    # No categorical features used in these models

    # Combine pipelines for different features into a single transformer
    pipeline = ColumnTransformer([
        ('number', number_transformer, number_features)
    ])

    # Fit the pipeline with the training data and return
    pipeline.fit(df_X)
    return pipeline


def get_target(df):
    """Get the target variable as a numpy array.""" 
    return df['target'].values