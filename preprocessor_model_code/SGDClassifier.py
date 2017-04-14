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

from sklearn.linear_model import SGDClassifier
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

    for (preprocessor, loss, penalty, alpha, learning_rate,
         fit_intercept, l1_ratio, eta0, power_t) in itertools.product(
                preprocessor_list,
                ['hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron'],
                ['l2', 'l1', 'elasticnet'],
                [0.000001, 0.00001, 0.0001, 0.001, 0.01],
                ['constant', 'optimal', 'invscaling'],
                [True, False],
                [0., 0.1, 0.15, 0.25, 0.5, 0.75, 0.9, 1.],
                [0.01, 0.1, 0.5, 1., 10., 50., 100.],
                [0., 0.1, 0.5, 1., 10., 50., 100.]):
        if penalty != 'elasticnet' and l1_ratio != 0.15:
            continue

        if learning_rate not in ['constant', 'invscaling'] and eta0 != 0.0:
            continue

        if learning_rate != 'invscaling' and power_t != 0.5:
            continue

        features = input_data.drop('class', axis=1).values.astype(float)
        labels = input_data['class'].values

        try:
            # Create the pipeline for the model
            clf = make_pipeline(preprocessor,
                                SGDClassifier(loss=loss,
                                              penalty=penalty,
                                              alpha=alpha,
                                              learning_rate=learning_rate,
                                              fit_intercept=fit_intercept,
                                              l1_ratio=l1_ratio,
                                              eta0=eta0,
                                              power_t=power_t,
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
        param_string += 'loss={},'.format(loss)
        param_string += 'penalty={},'.format(penalty)
        param_string += 'alpha={},'.format(alpha)
        param_string += 'learning_rate={},'.format(learning_rate)
        param_string += 'fit_intercept={},'.format(fit_intercept)
        param_string += 'l1_ratio={},'.format(l1_ratio)
        param_string += 'eta0={},'.format(eta0)
        param_string += 'power_t={}'.format(power_t)

        out_text = '\t'.join([dataset.split('/')[-1][:-7],
                              'SGDClassifier',
                              param_string,
                              str(accuracy),
                              str(macro_f1),
                              str(balanced_accuracy)])

        print(out_text)
        sys.stdout.flush()
