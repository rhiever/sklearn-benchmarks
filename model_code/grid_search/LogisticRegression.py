import sys
import pandas as pd
import numpy as np
import itertools
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from evaluate_model import evaluate_model

dataset = sys.argv[1]

pipeline_components = [RobustScaler, LogisticRegression]
pipeline_parameters = {}

C_values = np.arange(0.5, 20.1, 0.5)
penalty_values = ['l1', 'l2']
fit_intercept_values = [True, False]
dual_values = [True, False]
random_state = [324089]

all_param_combinations = itertools.product(C_values, penalty_values, fit_intercept_values, dual_values, random_state)
pipeline_parameters[LogisticRegression] = \
   [{'C': C, 'penalty': penalty, 'fit_intercept': fit_intercept, 'dual': dual, 'random_state': random_state}
     for (C, penalty, fit_intercept, dual, random_state) in all_param_combinations
     if not (penalty != 'l2' and dual != False)]

evaluate_model(dataset, pipeline_components, pipeline_parameters)
