import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import AdaBoostClassifier
from evaluate_model import evaluate_model

dataset = sys.argv[1]
num_param_combinations = int(sys.argv[2])
random_seed = int(sys.argv[3])

np.random.seed(random_seed)

pipeline_components = [RobustScaler, AdaBoostClassifier]
pipeline_parameters = {}

learning_rate_values = np.random.uniform(low=1e-10, high=5., size=num_param_combinations)
n_estimators_values = np.random.choice(list(range(50, 1001, 50)), size=num_param_combinations)

all_param_combinations = zip(learning_rate_values, n_estimators_values)
pipeline_parameters[AdaBoostClassifier] = [{'learning_rate': learning_rate, 'n_estimators': n_estimators, 'random_state': 324089}
                                    for (learning_rate, n_estimators) in all_param_combinations]

evaluate_model(dataset, pipeline_components, pipeline_parameters)
