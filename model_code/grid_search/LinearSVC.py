import sys
import pandas as pd
import numpy as np
import itertools
from sklearn.preprocessing import RobustScaler
from sklearn.svm import LinearSVC
from evaluate_model import evaluate_model

dataset = sys.argv[1]

pipeline_components = [RobustScaler, LinearSVC]
pipeline_parameters = {}

C_values = np.concatenate((np.arange(0., 1.0, 0.1), np.arange(1., 10.01, 1.)))
loss_values = ['hinge', 'squared_hinge']
penalty_values = ['l1', 'l2']
dual_values = [True, False]
fit_intercept_values = [True, False]
random_state = [324089]

all_param_combinations = itertools.product(C_values, loss_values, penalty_values, dual_values, fit_intercept_values, random_state)
pipeline_parameters[LinearSVC] = \
   [{'C': C, 'penalty': penalty, 'fit_intercept': fit_intercept, 'dual': dual, 'random_state': random_state}
     for (C, loss, penalty, dual, fit_intercept, random_state) in all_param_combinations]

evaluate_model(dataset, pipeline_components, pipeline_parameters)
