import sys
import pandas as pd
import numpy as np
import itertools
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import GradientBoostingClassifier
from evaluate_model import evaluate_model

dataset = sys.argv[1]

pipeline_components = [RobustScaler, GradientBoostingClassifier]
pipeline_parameters = {}

n_estimators_values = [10, 50, 100, 500]
min_impurity_decrease_values = np.arange(0., 0.005, 0.00025)
max_features_values = [0.1, 0.25, 0.5, 0.75, 'sqrt', 'log2', None]
learning_rate_values = [0.01, 0.1, 0.5, 1.0, 10.0, 50.0, 100.0]
loss_values = ['deviance', 'exponential']
random_state = [324089]

all_param_combinations = itertools.product(n_estimators_values, min_impurity_decrease_values, max_features_values, learning_rate_values, loss_values, random_state)
pipeline_parameters[GradientBoostingClassifier] = \
   [{'n_estimators': n_estimators, 'min_impurity_decrease': min_impurity_decrease, 'max_features': max_features, 'learning_rate': learning_rate, 'loss': loss, 'random_state': random_state}
     for (n_estimators, min_impurity_decrease, max_features, learning_rate, loss, random_state) in all_param_combinations]

evaluate_model(dataset, pipeline_components, pipeline_parameters)
