import sys
import pandas as pd
import numpy as np
import itertools
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.pipeline import make_pipeline
import itertools

dataset = sys.argv[1]

# Read the data set into memory
input_data = pd.read_csv(dataset, compression='gzip', sep='\t').sample(frac=1., replace=False, random_state=42)

for (C, loss, fit_intercept) in itertools.product([0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 0.5, 1., 10., 50., 100.],
                                                  ['hinge', 'squared_hinge'],
                                                  [True, False]):
    features = input_data.drop('class', axis=1).values.astype(float)
    labels = input_data['class'].values

    try:
        # Create the pipeline for the model
        clf = make_pipeline(StandardScaler(),
                            PassiveAggressiveClassifier(C=C,
                                                        loss=loss,
                                                        fit_intercept=fit_intercept,
                                                        random_state=324089))
        # 10-fold CV scores for the pipeline
        cv_scores = cross_val_score(estimator=clf, X=features, y=labels, cv=10)
    except KeyboardInterrupt:
        sys.exit(1)
    except:
        continue

    param_string = ''
    param_string += 'C={},'.format(C)
    param_string += 'loss={},'.format(loss)
    param_string += 'fit_intercept={}'.format(fit_intercept)

    for cv_score in cv_scores:
        out_text = '\t'.join([dataset.split('/')[-1][:-7],
                              'PassiveAggressiveClassifier',
                              param_string,
                              str(cv_score)])

        print(out_text)
        sys.stdout.flush()
