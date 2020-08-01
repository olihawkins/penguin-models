# -*- coding: utf-8 -*-

"""A module for plotting penguins data for modelling with scikit-learn."""

# Imports ---------------------------------------------------------------------

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Constants -------------------------------------------------------------------

SPECIES_COLORS = {
    'Adelie': '#4daf4a', 
    'Gentoo': '#ffb000', 
    'Chinstrap': '#0084f7'
}

X_AXIS = [30, 60]
Y_AXIS = [12, 22]

# Set style -------------------------------------------------------------------

# Load the style from a file
plt.style.use('./style/eda.mplstyle')

# Alternatively, load the style from the library in ~/.matplotlib/stylelib
# plt.style.use(['eda'])

# Functions -------------------------------------------------------------------

def get_contour_data(model, pipeline, n_points=1000):

    """Create the data used to show the boundary of the decision function."""
    
    x0s = np.linspace(X_AXIS[0], X_AXIS[1], n_points)
    x1s = np.linspace(Y_AXIS[0], Y_AXIS[1], n_points)
    x0, x1 = np.meshgrid(x0s, x1s)
    X = np.c_[x0.ravel(), x1.ravel()]
    df_X = pd.DataFrame(X, columns=['bill_length_mm', 'bill_depth_mm'])
    X = pipeline.transform(df_X)
    y_pred = model.predict(X).reshape(x0.shape)
    y_decision = model.decision_function(X).reshape(x0.shape)
    return x0, x1, y_pred, y_decision


def get_target_colors(target):

    """Create a dictionary of colors to use in binary classification plots."""
    
    return {
        target : '#984ea3',
        'Other': '#ff7f00'
    }

# Plots -----------------------------------------------------------------------

def plot_example():

    plt.style.reload_library()
    plt.style.use(['eda'])
    fig, ax = plt.subplots()
    ax.set_title('Some random words of the title')
    ax.scatter(np.random.normal(0,1,10), np.random.normal(0,1,10))
    fig.savefig('plots/test.svg', format='svg')
    fig.savefig('plots/test.png', format='png')
    plt.close()


def plot_target_by_features(df):

    """Plot the different target species."""

    fig, ax = plt.subplots()

    ax.set_title(
        label='Palmer penguins by species and bill characteristics', 
        loc='center')

    ax.get_xaxis().set_major_formatter(
        mpl.ticker.FormatStrFormatter('%.0f'))
    ax.set_xlim(X_AXIS[0], X_AXIS[1])
    ax.set_xlabel('Bill length (mm)')

    ax.get_yaxis().set_major_formatter(
        mpl.ticker.FormatStrFormatter('%.0f'))
    ax.set_ylim(Y_AXIS[0], Y_AXIS[1])
    ax.set_ylabel('Bill depth (mm)')
    
    grouped = df.groupby('species')
    for key, group in grouped:
        ax.scatter(
            group['bill_length_mm'], 
            group['bill_depth_mm'], 
            c=SPECIES_COLORS[key],
            s=40,
            label=key,
            alpha=0.55)
    
    ax.legend(loc='lower left', handletextpad=0.2)
    fig.savefig('plots/target-by-features.png', format='png')
    plt.close()


def plot_model(df, model, pipeline, f_score, target, title, filename):

    """Plot the results of a binary classification model."""

    fig, ax = plt.subplots()

    ax.set_title(title, loc='center')

    ax.get_xaxis().set_major_formatter(
        mpl.ticker.FormatStrFormatter('%.0f'))
    ax.set_xlim(X_AXIS[0], X_AXIS[1])
    ax.set_xlabel('Bill length (mm)')

    ax.get_yaxis().set_major_formatter(
        mpl.ticker.FormatStrFormatter('%.0f'))
    ax.set_ylim(Y_AXIS[0], Y_AXIS[1])
    ax.set_ylabel('Bill depth (mm)')

    # Plot the boundary of the decision function
    x0, x1, y_pred, y_decision = get_contour_data(model, pipeline)
    ax.contourf(x0, x1, y_pred, cmap=plt.cm.PuOr, alpha=0.2)

    # This plots the decision score, if needed
    # ax.contourf(x0, x1, y_decision, cmap=plt.cm.PuOr, alpha=0.1)

    df = df.copy()
    df['species'] = df['target'].apply(lambda t: target if t == 1 else 'Other')

    colors = get_target_colors(target)

    grouped = df.groupby('species')
    for key, group in grouped:
        ax.scatter(
            group['bill_length_mm'], 
            group['bill_depth_mm'], 
            c=colors[key],
            s=40,
            label=key, 
            alpha=0.55)
    
    ax.legend(loc='lower left', handletextpad=0.2)

    bbox_style = {
        'boxstyle': 'round', 
        'facecolor': '#ffffff', 
        'edgecolor': '#d4d4d4', 
        'alpha': 0.8
    }

    ax.text(53, 12.415, '$F_1$ score: {0}'.format(f_score), bbox=bbox_style)
    
    fig.savefig('plots/{0}.png'.format(filename), format='png')
    plt.close()