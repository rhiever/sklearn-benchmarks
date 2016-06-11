import sys
import pandas as pd
import numpy as np
import itertools
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import cross_val_score
from sklearn.pipeline import make_pipeline
import itertools

dataset = sys.argv[1]

# Read the data set into memory
input_data = pd.read_csv(dataset, compression='gzip', sep='\t')

for (C, penalty, fit_intercept, dual, tol) in itertools.product([0.01, 0.1, 0.5,1.0, 10.0, 50.0, 100.0],
                                                                ['l1', 'l2'],
                                                                [True, False],
                                                                [True, False],
                                                                [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]):
    if penalty != 'l2' and dual != False:
        continue

    features = input_data.drop('class', axis=1).values.astype(float)
    labels = input_data['class'].values

    try:
        # Create the pipeline for the model
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(C=C,
                                               penalty=penalty,
                                               fit_intercept=fit_intercept,
                                               dual=dual,
                                               tol=tol,
                                               random_state=324089))
        # 10-fold CV scores for the pipeline
        cv_scores = cross_val_score(estimator=clf, X=features, y=labels, cv=10)
    except KeyboardInterrupt:
        sys.exit(1)
    except:
        continue

    param_string = ''
    param_string += 'C={},'.format(C)
    param_string += 'penalty={},'.format(penalty)
    param_string += 'fit_intercept={},'.format(fit_intercept)
    param_string += 'dual={},'.format(dual)
    param_string += 'tol={}'.format(tol)

    for cv_score in cv_scores:
        out_text = '\t'.join([dataset.split('/')[-1][:-7],
                              'LogisticRegression',
                              param_string,
                              str(cv_score)])

        print(out_text)
        sys.stdout.flush()
