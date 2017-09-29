import sys
import pandas as pd
import numpy as np
import itertools
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVC
from evaluate_model import evaluate_model

dataset = sys.argv[1]

pipeline_components = [RobustScaler, SVC]
pipeline_parameters = {}

C_values = [0.01, 0.1, 0.5, 1., 10., 50., 100.]
gamma_values = [0.01, 0.1, 0.5, 1., 10., 50., 100., 'auto']
kernel_values = ['poly', 'rbf', 'sigmoid']
degree_values = [2, 3]
coef0_values = [0., 0.1, 0.5, 1., 10., 50., 100.]
random_state = [324089]

all_param_combinations = itertools.product(C_values, gamma_values, kernel_values, degree_values, coef0_values, random_state)
pipeline_parameters[SVC] = \
   [{'C': C, 'gamma': gamma, 'kernel': kernel, 'degree': degree, 'coef0': coef0, 'random_state': random_state}
     for (C, gamma, kernel, degree, coef0, random_state) in all_param_combinations
     if not (kernel != 'poly' and degree > 2) and not (kernel not in ['poly', 'sigmoid'] and coef0 != 0.0)]

evaluate_model(dataset, pipeline_components, pipeline_parameters)
