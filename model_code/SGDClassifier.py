import sys
import pandas as pd
import numpy as np
import itertools
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.pipeline import make_pipeline
import itertools

dataset = sys.argv[1]

# Read the data set into memory
input_data = pd.read_csv(dataset, compression='gzip', sep='\t')

for (loss, penalty, alpha, learning_rate, fit_intercept, l1_ratio, eta0, power_t) in itertools.product(['hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron'],
                                                                                                                ['l2', 'l1', 'elasticnet'],
                                                                                                                [0.000001, 0.00001, 0.0001, 0.001, 0.01],
                                                                                                                ['constant', 'optimal', 'invscaling'],
                                                                                                                [True, False],
                                                                                                                [0.0, 0.1, 0.15, 0.25, 0.5, 0.75, 0.9, 1.0],
                                                                                                                [0.01, 0.1, 0.5, 1.0, 10.0, 50.0, 100.0],
                                                                                                                [0.0, 0.1, 0.5, 1.0, 10.0, 50.0, 100.0]):
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
        clf = make_pipeline(StandardScaler(),
                            SGDClassifier(loss=loss,
                                          penalty=penalty,
                                          alpha=alpha,
                                          learning_rate=learning_rate,
                                          fit_intercept=fit_intercept,
                                          l1_ratio=l1_ratio,
                                          eta0=eta0,
                                          power_t=power_t,
                                          random_state=324089))
        # 10-fold CV scores for the pipeline
        cv_scores = cross_val_score(estimator=clf, X=features, y=labels, cv=10)
    except KeyboardInterrupt:
        sys.exit(1)
    except:
        continue

    param_string = ''
    param_string += 'loss={},'.format(loss)
    param_string += 'penalty={},'.format(penalty)
    param_string += 'alpha={},'.format(alpha)
    param_string += 'learning_rate={},'.format(learning_rate)
    param_string += 'fit_intercept={},'.format(fit_intercept)
    param_string += 'l1_ratio={},'.format(l1_ratio)
    param_string += 'eta0={},'.format(eta0)
    param_string += 'power_t={}'.format(power_t)

    for cv_score in cv_scores:
        out_text = '\t'.join([dataset.split('/')[-1][:-7],
                              'SGDClassifier',
                              param_string,
                              str(cv_score)])

        print(out_text)
        sys.stdout.flush()
