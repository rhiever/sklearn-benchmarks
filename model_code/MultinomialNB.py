import sys
import pandas as pd
import numpy as np
import itertools
from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import MultinomialNB
from evaluate_model import evaluate_model

dataset = sys.argv[1]

pipeline_components = [MinMaxScaler, MultinomialNB]
pipeline_parameters = {}

alpha_values = [0., 0.1, 0.25, 0.5, 0.75, 1., 5., 10., 25., 50.]
fit_prior_values = [True, False]

all_param_combinations = itertools.product(alpha_values, fit_prior_values)
pipeline_parameters[MultinomialNB] = \
   [{'alpha': alpha, 'fit_prior': fit_prior}
     for (alpha, fit_prior) in all_param_combinations]

evaluate_model(dataset, pipeline_components, pipeline_parameters)
