import sys
import pandas as pd
import numpy as np
import itertools
from sklearn.preprocessing import RobustScaler
from sklearn.naive_bayes import GaussianNB
from evaluate_model import evaluate_model

dataset = sys.argv[1]

pipeline_components = [RobustScaler, GaussianNB]
pipeline_parameters = {}
pipeline_parameters[GaussianNB] = [{}]

evaluate_model(dataset, pipeline_components, pipeline_parameters)
