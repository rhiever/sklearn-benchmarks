import sys
import pandas as pd
import numpy as np
import itertools
from sklearn.preprocessing import RobustScaler
from sklearn.neighbors import KNeighborsClassifier
from evaluate_model import evaluate_model

dataset = sys.argv[1]

pipeline_components = [RobustScaler, KNeighborsClassifier]
pipeline_parameters = {}

n_neighbors_values = list(range(1, 26)) + [50, 100]
weights_values = ['uniform', 'distance']

all_param_combinations = itertools.product(n_neighbors_values, weights_values)
pipeline_parameters[KNeighborsClassifier] = \
   [{'n_neighbors': n_neighbors, 'weights': weights}
     for (n_neighbors, weights) in all_param_combinations]

evaluate_model(dataset, pipeline_components, pipeline_parameters)
