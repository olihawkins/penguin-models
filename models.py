# -*- coding: utf-8 -*-

"""A module for building and testing models using penguin data."""

# Imports ---------------------------------------------------------------------

import data
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.svm import SVC

from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

# Constants -------------------------------------------------------------------
 
RANDOM_STATE = data.RANDOM_STATE

# Models ----------------------------------------------------------------------

def get_logistic_regression(X_train, y_train):

    model = LogisticRegression(random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    return model


def get_linear_svc(X_train, y_train):

    model = LinearSVC(
        C=1.0,
        loss='hinge',
        random_state=RANDOM_STATE,
        max_iter=10000)

    model.fit(X_train, y_train)
    return model


def get_svc(X_train, y_train, polynomial_degree=3):

    model = SVC(
        kernel='poly',
        degree=polynomial_degree,
        coef0=1,
        C=5,
        random_state=RANDOM_STATE,
        max_iter=10000)

    model.fit(X_train, y_train)
    return model


def get_sgd_classifier(X_train, y_train):

    model = SGDClassifier(random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    return model


def get_random_forest_classifier(X_train, y_train):

    model = RandomForestClassifier(
        class_weight='balanced_subsample',
        random_state=RANDOM_STATE)

    model.fit(X_train, y_train)
    return model

# Training --------------------------------------------------------------------

def train_logistic_regression(
    target_species,
    add_polynomials=False, 
    polynomial_degree=3):

    df = data.load_train_set(target_species=target_species)
    df_X = df[['bill_length_mm', 'bill_depth_mm']]
    pipeline = data.get_pipeline(df_X, add_polynomials, polynomial_degree)
    X_train = pipeline.transform(df_X)
    y_train = data.get_target(df)
    model = get_logistic_regression(X_train, y_train)

    return df, model, pipeline, X_train, y_train


def train_linear_svc(
    target_species,
    add_polynomials=False, 
    polynomial_degree=3):

    df = data.load_train_set(target_species=target_species)
    df_X = df[['bill_length_mm', 'bill_depth_mm']]
    pipeline = data.get_pipeline(df_X, add_polynomials, polynomial_degree)
    X_train = pipeline.transform(df_X)
    y_train = data.get_target(df)
    model = get_linear_svc(X_train, y_train)

    return df, model, pipeline, X_train, y_train


def train_polynomial_svc(
    target_species, 
    polynomial_degree=3):

    df = data.load_train_set(target_species=target_species)
    df_X = df[['bill_length_mm', 'bill_depth_mm']]
    pipeline = data.get_pipeline(df_X)
    X_train = pipeline.transform(df_X)
    y_train = data.get_target(df)
    model = get_svc(X_train, y_train, polynomial_degree)

    return df, model, pipeline, X_train, y_train


# Evaluation ------------------------------------------------------------------

def evaluate(model, X_train, y_train):
    y_train_pred = cross_val_predict(model, X_train, y_train, cv=5)
    cm = confusion_matrix(y_train, y_train_pred)
    f_score = f1_score(y_train, y_train_pred)
    return y_train_pred, cm, f_score


def get_f1_score(model, X_train, y_train):
    y_train_pred = cross_val_predict(model, X_train, y_train, cv=5)
    f_score = f1_score(y_train, y_train_pred)
    return f_score