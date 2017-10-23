import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from evaluate_model import evaluate_model

dataset = sys.argv[1]
num_param_combinations = int(sys.argv[2])
random_seed = int(sys.argv[3])

np.random.seed(random_seed)

pipeline_components = [RobustScaler, LogisticRegression]
pipeline_parameters = {}

C_values = np.random.uniform(low=1e-10, high=10., size=num_param_combinations)
penalty_values = np.random.choice(['l1', 'l2'], size=num_param_combinations)
fit_intercept_values = np.random.choice([True, False], size=num_param_combinations)
dual_values = np.random.choice([True, False], size=num_param_combinations)

all_param_combinations = zip(C_values, penalty_values, fit_intercept_values, dual_values)
pipeline_parameters[LogisticRegression] = \
   [{'C': C, 'penalty': penalty, 'fit_intercept': fit_intercept, 'dual': False if penalty != 'l2' else dual, 'random_state': 324089}
     for (C, penalty, fit_intercept, dual) in all_param_combinations]

evaluate_model(dataset, pipeline_components, pipeline_parameters)
