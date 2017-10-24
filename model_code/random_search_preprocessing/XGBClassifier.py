import sys
import pandas as pd
import numpy as np

from sklearn.preprocessing import Binarizer, MaxAbsScaler, MinMaxScaler
from sklearn.preprocessing import Normalizer, PolynomialFeatures, RobustScaler, StandardScaler
from sklearn.decomposition import FastICA, PCA
from sklearn.kernel_approximation import RBFSampler, Nystroem
from sklearn.cluster import FeatureAgglomeration
from sklearn.feature_selection import SelectFwe, SelectPercentile, VarianceThreshold
from sklearn.feature_selection import SelectFromModel, RFE
from sklearn.ensemble import ExtraTreesClassifier

from xgboost import XGBClassifier # Assumes XGBoost v0.6
from evaluate_model import evaluate_model

dataset = sys.argv[1]
num_param_combinations = int(sys.argv[2])
random_seed = int(sys.argv[3])
preprocessor_num = int(sys.argv[4])

np.random.seed(random_seed)

preprocessor_list = [Binarizer, MaxAbsScaler, MinMaxScaler, Normalizer,
                     PolynomialFeatures, RobustScaler, StandardScaler,
                     FastICA, PCA, RBFSampler, Nystroem, FeatureAgglomeration,
                     SelectFwe, SelectPercentile, VarianceThreshold,
                     SelectFromModel, RFE]

chosen_preprocessor = preprocessor_list[preprocessor_num]

pipeline_components = [chosen_preprocessor, XGBClassifier]
pipeline_parameters = {}

n_estimators_values = np.random.choice(list(range(50, 1001, 50)), size=num_param_combinations)
learning_rate_values = np.random.uniform(low=1e-10, high=5., size=num_param_combinations)
gamma_values = np.random.uniform(low=0., high=1., size=num_param_combinations)
max_depth_values = np.random.choice(list(range(1, 51)) + [None], size=num_param_combinations)
subsample_values = np.random.uniform(low=0., high=1., size=num_param_combinations)

all_param_combinations = zip(n_estimators_values, learning_rate_values, gamma_values, max_depth_values, subsample_values)
pipeline_parameters[XGBClassifier] = \
   [{'n_estimators': n_estimators, 'learning_rate': learning_rate, 'gamma': gamma, 'max_depth': max_depth, 'subsample': subsample, 'seed': 324089, 'nthread': 1}
     for (n_estimators, learning_rate, gamma, max_depth, subsample) in all_param_combinations]

if chosen_preprocessor is SelectFromModel:
    pipeline_parameters[SelectFromModel] = [{'estimator': ExtraTreesClassifier(n_estimators=100, random_state=324089)}]
elif chosen_preprocessor is RFE:
    pipeline_parameters[RFE] = [{'estimator': ExtraTreesClassifier(n_estimators=100, random_state=324089)}]

evaluate_model(dataset, pipeline_components, pipeline_parameters)
