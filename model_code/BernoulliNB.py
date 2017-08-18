import sys
import pandas as pd
import numpy as np
import itertools
from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import BernoulliNB
from evaluate_model import evaluate_model

dataset = sys.argv[1]

pipeline_components = [MinMaxScaler, BernoulliNB]
pipeline_parameters = {}

alpha_values = [0., 0.1, 0.25, 0.5, 0.75, 1., 5., 10., 25., 50.]
fit_prior_values = [True, False]
binarize_values = [0., 0.1, 0.25, 0.5, 0.75, 0.9, 1.]
pipeline_parameters[BernoulliNB] = [{'alpha': args[0], 'fit_prior': args[1], 'binarize': args[2]}
                                    for args in itertools.product(alpha_values, fit_prior_values, binarize_values)]

evaluate_model(dataset, pipeline_components, pipeline_parameters)
