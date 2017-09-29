import sys
import pandas as pd
import numpy as np
import itertools
from sklearn.preprocessing import RobustScaler
from sklearn.tree import DecisionTreeClassifier
from evaluate_model import evaluate_model

dataset = sys.argv[1]

pipeline_components = [RobustScaler, DecisionTreeClassifier]
pipeline_parameters = {}

min_impurity_decrease_values = np.arange(0., 0.005, 0.00025)
max_features_values = [0.1, 0.25, 0.5, 0.75, 'sqrt', 'log2', None]
criterion_values = ['gini', 'entropy']
random_state = [324089]

all_param_combinations = itertools.product(min_impurity_decrease_values, max_features_values, criterion_values, random_state)
pipeline_parameters[DecisionTreeClassifier] = \
   [{'min_impurity_decrease': min_impurity_decrease, 'max_features': max_features, 'criterion': criterion, 'random_state': random_state}
     for (min_impurity_decrease, max_features, criterion, random_state) in all_param_combinations]

evaluate_model(dataset, pipeline_components, pipeline_parameters)
