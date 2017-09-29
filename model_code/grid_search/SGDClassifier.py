import sys
import pandas as pd
import numpy as np
import itertools
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import SGDClassifier
from evaluate_model import evaluate_model

dataset = sys.argv[1]

pipeline_components = [RobustScaler, SGDClassifier]
pipeline_parameters = {}

loss_values = ['hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron']
penalty_values = ['l2', 'l1', 'elasticnet']
alpha_values = [0.000001, 0.00001, 0.0001, 0.001, 0.01]
learning_rate_values = ['constant', 'optimal', 'invscaling']
fit_intercept_values = [True, False]
l1_ratio_values = [0., 0.1, 0.15, 0.25, 0.5, 0.75, 0.9, 1.]
eta0_values = [0.0, 0.01, 0.1, 0.5, 1., 10., 50., 100.]
power_t_values = [0., 0.1, 0.5, 1., 10., 50., 100.]
random_state = [324089]

all_param_combinations = itertools.product(loss_values, penalty_values, alpha_values, learning_rate_values, fit_intercept_values, l1_ratio_values, eta0_values, power_t_values, random_state)
pipeline_parameters[SGDClassifier] = \
   [{'loss': loss, 'penalty': penalty, 'alpha': alpha, 'learning_rate': learning_rate, 'fit_intercept': fit_intercept, 'l1_ratio': l1_ratio, 'eta0': eta0, 'power_t': power_t, 'random_state': random_state}
     for (loss, penalty, alpha, learning_rate, fit_intercept, l1_ratio, eta0, power_t, random_state) in all_param_combinations
     if not (penalty != 'elasticnet' and l1_ratio != 0.15) and not (learning_rate not in ['constant', 'invscaling'] and eta0 != 0.0) and not (learning_rate != 'invscaling' and power_t != 0.5)]

evaluate_model(dataset, pipeline_components, pipeline_parameters)
