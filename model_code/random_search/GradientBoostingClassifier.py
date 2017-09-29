import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import GradientBoostingClassifier
from evaluate_model import evaluate_model

dataset = sys.argv[1]
num_param_combinations = int(sys.argv[2])
random_seed = int(sys.argv[3])

np.random.seed(random_seed)

pipeline_components = [RobustScaler, GradientBoostingClassifier]
pipeline_parameters = {}

n_estimators_values = np.random.choice(list(range(50, 1001, 50)), size=num_param_combinations)
min_impurity_decrease_values = np.random.exponential(scale=0.01, size=num_param_combinations)
max_features_values = np.random.choice(list(np.arange(0.01, 1., 0.01)) + ['sqrt', 'log2', None], size=num_param_combinations)
learning_rate_values = np.random.uniform(low=1e-10, high=5., size=num_param_combinations)
loss_values = np.random.choice(['deviance', 'exponential'], size=num_param_combinations)
max_depth_values = np.random.choice(list(range(1, 51)) + [None], size=num_param_combinations)

all_param_combinations = zip(n_estimators_values, min_impurity_decrease_values, max_features_values, learning_rate_values, loss_values, max_depth_values)
pipeline_parameters[GradientBoostingClassifier] = \
   [{'n_estimators': n_estimators, 'min_impurity_decrease': min_impurity_decrease, 'max_features': max_features, 'learning_rate': learning_rate, 'loss': loss, 'max_depth': max_depth, 'random_state': 324089}
     for (n_estimators, min_impurity_decrease, max_features, learning_rate, loss, max_depth) in all_param_combinations]

evaluate_model(dataset, pipeline_components, pipeline_parameters)
