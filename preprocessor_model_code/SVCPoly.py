import sys
import pandas as pd
import numpy as np
import itertools
import warnings

from sklearn.preprocessing import Binarizer, MaxAbsScaler, MinMaxScaler
from sklearn.preprocessing import Normalizer, PolynomialFeatures, RobustScaler, StandardScaler
from sklearn.decomposition import FastICA, PCA
from sklearn.kernel_approximation import RBFSampler, Nystroem
from sklearn.cluster import FeatureAgglomeration
from sklearn.feature_selection import SelectFwe, SelectKBest, SelectPercentile, VarianceThreshold
from sklearn.feature_selection import SelectFromModel, RFE
from sklearn.ensemble import ExtraTreesClassifier

from sklearn.svm import SVC
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score, f1_score
from tpot_metrics import balanced_accuracy_score
from sklearn.pipeline import make_pipeline
import itertools

dataset = sys.argv[1]

preprocessor_list = [Binarizer(), MaxAbsScaler(), MinMaxScaler(), Normalizer(),
                     PolynomialFeatures(), RobustScaler(), StandardScaler(),
                     FastICA(), PCA(), RBFSampler(), Nystroem(), FeatureAgglomeration(),
                     SelectFwe(), SelectKBest(), SelectPercentile(), VarianceThreshold(),
                     SelectFromModel(estimator=ExtraTreesClassifier(n_estimators=100)),
                     RFE(estimator=ExtraTreesClassifier(n_estimators=100))]

# Read the data set into memory
input_data = pd.read_csv(dataset, compression='gzip', sep='\t').sample(frac=1., replace=False, random_state=42)

with warnings.catch_warnings():
    warnings.simplefilter('ignore')

    for (preprocessor, C, gamma, degree, coef0) in itertools.product(
                preprocessor_list,
                [0.01, 0.1, 0.5, 1., 10., 50., 100.],
                [0.01, 0.1, 0.5, 1., 10., 50., 100., 'auto'],
                [2, 3],
                [0., 0.1, 0.5, 1., 10., 50., 100.]):
        features = input_data.drop('class', axis=1).values.astype(float)
        labels = input_data['class'].values

        try:
            # Create the pipeline for the model
            clf = make_pipeline(preprocessor,
                                SVC(C=C,
                                    gamma=gamma,
                                    kernel='poly',
                                    degree=degree,
                                    coef0=coef0,
                                    random_state=324089))
            # 10-fold CV score for the pipeline
            cv_predictions = cross_val_predict(estimator=clf, X=features, y=labels, cv=10)
            accuracy = accuracy_score(labels, cv_predictions)
            macro_f1 = f1_score(labels, cv_predictions, average='macro')
            balanced_accuracy = balanced_accuracy_score(labels, cv_predictions)
        except KeyboardInterrupt:
            sys.exit(1)
        except:
            continue

        param_string = ''
        param_string += 'preprocessor={},'.format(preprocessor.__class__.__name__)
        param_string += 'C={},'.format(C)
        param_string += 'gamma={},'.format(gamma)
        param_string += 'kernel=poly,'
        param_string += 'degree={},'.format(degree)
        param_string += 'coef0={},'.format(coef0)

        out_text = '\t'.join([dataset.split('/')[-1][:-7],
                              'SVC',
                              param_string,
                              str(accuracy),
                              str(macro_f1),
                              str(balanced_accuracy)])

        print(out_text)
        sys.stdout.flush()
