import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.neighbors import KNeighborsClassifier
from evaluate_model import evaluate_model

dataset = sys.argv[1]
num_param_combinations = int(sys.argv[2])
random_seed = int(sys.argv[3])

np.random.seed(random_seed)

pipeline_components = [RobustScaler, KNeighborsClassifier]
pipeline_parameters = {}

n_neighbors_values = np.random.randint(low=1, high=100, size=num_param_combinations)
weights_values = np.random.choice(['uniform', 'distance'], size=num_param_combinations)

all_param_combinations = zip(n_neighbors_values, weights_values)
pipeline_parameters[KNeighborsClassifier] = \
   [{'n_neighbors': n_neighbors, 'weights': weights}
     for (n_neighbors, weights) in all_param_combinations]

evaluate_model(dataset, pipeline_components, pipeline_parameters)
