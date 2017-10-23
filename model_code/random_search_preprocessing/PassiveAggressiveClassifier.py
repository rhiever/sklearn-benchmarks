import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import PassiveAggressiveClassifier
from evaluate_model import evaluate_model

dataset = sys.argv[1]
num_param_combinations = int(sys.argv[2])
random_seed = int(sys.argv[3])

np.random.seed(random_seed)

pipeline_components = [RobustScaler, PassiveAggressiveClassifier]
pipeline_parameters = {}

C_values = np.random.uniform(low=1e-10, high=10., size=num_param_combinations)
loss_values = np.random.choice(['hinge', 'squared_hinge'], size=num_param_combinations)
fit_intercept_values = np.random.choice([True, False], size=num_param_combinations)

all_param_combinations = zip(C_values, loss_values, fit_intercept_values)
pipeline_parameters[PassiveAggressiveClassifier] = \
   [{'C': C, 'loss': loss, 'fit_intercept': fit_intercept, 'random_state': 324089}
     for (C, loss, fit_intercept) in all_param_combinations]

evaluate_model(dataset, pipeline_components, pipeline_parameters)
