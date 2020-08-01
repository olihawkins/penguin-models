# -*- coding: utf-8 -*-

"""A module for analysing penguins data for modelling with scikit-learn."""

# Imports ---------------------------------------------------------------------

import data
import models
import plots

from sklearn.model_selection import cross_val_predict
from sklearn.metrics import f1_score

# Functions -------------------------------------------------------------------

def targets():

    """Plot the targets from the training set."""

    df = data.load_train_set()
    plots.plot_target_by_features(df)


def logistic_regression(target):

    """
    Train a logistic regression and plot its cross-validated results.
    """

    # Get data, pipeline and model
    df, model, pipeline, X_train, y_train = \
        models.train_logistic_regression(target_species=target)

    # Get f-score
    y_train_pred = cross_val_predict(model, X_train, y_train, cv=5)
    f_score = '{:.3f}'.format(f1_score(y_train, y_train_pred))

    # Plot
    plots.plot_model(
        df, 
        model, 
        pipeline,
        f_score, 
        target,
        '{} penguins classified with logistic regression'.format(target),
        'logistic-regression-{}'.format(target.lower()))


def linear_svc(target):

    """
    Train a linear SVC and plot its cross-validated results.
    """

    # Get data, pipeline and model
    df, model, pipeline, X_train, y_train = \
        models.train_linear_svc(target_species=target)

    # Get f-score
    y_train_pred = cross_val_predict(model, X_train, y_train, cv=5)
    f_score = '{:.3f}'.format(f1_score(y_train, y_train_pred))

    # Plot
    plots.plot_model(
        df, 
        model, 
        pipeline,
        f_score, 
        target,
        '{} penguins classified with linear SVM'.format(target),
        'linear-svm-{}'.format(target.lower()))


def polynomial_svc(target):

    """
    Train a polynomial SVC and plot its cross-validated results.
    """

    # Get data, pipeline and model
    df, model, pipeline, X_train, y_train = \
        models.train_polynomial_svc(target_species=target)

    # Get f-score
    y_train_pred = cross_val_predict(model, X_train, y_train, cv=5)
    f_score = '{:.3f}'.format(f1_score(y_train, y_train_pred))

    # Plot
    plots.plot_model(
        df, 
        model, 
        pipeline,
        f_score, 
        target,
        '{} penguins classified with polynomial SVM'.format(target),
        'polynomial-svm-{}'.format(target.lower()))


def polynomial_svc_test(target):

    """
    Train a polynomial SVC, predict the results for the test set, and plot 
    the results.
    """

    # Get data, pipeline and model
    df, model, pipeline, X_train, y_train = \
        models.train_polynomial_svc(target_species=target)

    # Load the test set
    df = data.load_test_set(target_species=target)

    # Get f-score
    df_X = df[['bill_length_mm', 'bill_depth_mm']]
    X_test = pipeline.transform(df_X)
    y_test = data.get_target(df)
    y_test_pred = model.predict(X_test)
    f_score = '{:.3f}'.format(f1_score(y_test, y_test_pred))

    # Plot
    plots.plot_model(
        df, 
        model, 
        pipeline,
        f_score, 
        target,
        '{}s in test set classified with trained polynomial SVM'.format(target),
        'polynomial-svm-test-{}'.format(target.lower()))

# Analysis --------------------------------------------------------------------

def run():

    # Plot targets
    targets()

    # Fit and plot logistic regressions
    logistic_regression('Adelie')
    logistic_regression('Gentoo')
    logistic_regression('Chinstrap')

    # Fit and plot linear SVM classifiers
    linear_svc('Adelie')
    linear_svc('Gentoo')
    linear_svc('Chinstrap')


    # Fit and plot polynomial SVM classifiers
    polynomial_svc('Adelie')
    polynomial_svc('Gentoo')
    polynomial_svc('Chinstrap')

    # Fit and plot polynomial SVM classifier against the test set
    polynomial_svc_test('Chinstrap')