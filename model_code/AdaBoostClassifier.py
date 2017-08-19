import sys
import pandas as pd
import numpy as np
import itertools
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import AdaBoostClassifier
from evaluate_model import evaluate_model

dataset = sys.argv[1]

pipeline_components = [RobustScaler, AdaBoostClassifier]
pipeline_parameters = {}

learning_rate_values = [0.01, 0.1, 0.5, 1.0, 10.0, 50.0, 100.0]
n_estimators_values = [10, 50, 100, 500]
random_state = [324089]

all_param_combinations = itertools.product(learning_rate_values, n_estimators_values, random_state)
pipeline_parameters[AdaBoostClassifier] = [{'learning_rate': learning_rate, 'n_estimators': n_estimators, 'random_state': random_state}
                                    for (learning_rate, n_estimators, random_state) in all_param_combinations]

evaluate_model(dataset, pipeline_components, pipeline_parameters)
