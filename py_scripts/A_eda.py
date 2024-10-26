# Functions and variables in A_eda.ipynb that will be exported to B_preprocess.ipynb

# Importing libraries
from pathlib import Path

import pandas as pd
import numpy as np
import math

import matplotlib.pyplot as plt
import seaborn as sns

from IPython.display import display

# Setting seed
np.random.seed(42)


# SAVE_FIG: a function to save any visualizations in the "plots" folder
IMAGES_PATH = Path() / 'plots'

def save_fig(fig_name, tight_layout=True, fig_extension='png', resolution=300):
    path = IMAGES_PATH / f'{fig_name}.{fig_extension}'
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)


# IMPORTANT VARIABLES
# Importing data
adclicks = pd.read_csv('data/ad_click_dataset.csv')
# Dropping `full_name`
adclicks = adclicks.drop('full_name', axis=1, inplace=False)

# Select the categorical features
categoricals = list(adclicks.select_dtypes(include=['category', 'object']).columns)


# Splitting our data
from sklearn.model_selection import StratifiedShuffleSplit

# Generating 10 different splits of 20/80 for test/train
splitter = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=42)
strat_splits = []
for train_index, test_index in splitter.split(adclicks, adclicks['click']):
    strat_train_set_n = adclicks.iloc[train_index]
    strat_test_set_n = adclicks.iloc[test_index]
    strat_splits.append([strat_train_set_n, strat_test_set_n])
    
# We will be using the first split of our stratified splits
strat_train_set, strat_test_set = strat_splits[0]

# Make a copy of the training set incase we want to revert
adclicks = strat_train_set.copy()


# PLT_SHOW: a function to name and save the image of the current plot
def plt_show(plt_name):
    save_fig(plt_name)
    plt.show()
