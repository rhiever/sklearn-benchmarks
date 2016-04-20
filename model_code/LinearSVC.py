import sys
import pandas as pd
import itertools
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.cross_validation import cross_val_score
from sklearn.pipeline import make_pipeline
import itertools

dataset = sys.argv[1]

# Read the data set into memory
input_data = pd.read_csv(dataset, compression='gzip', sep='\t')

for (C, loss, penalty, dual, tol, fit_intercept) in itertools.product([0.01, 0.1, 0.5, 1.0, 10.0, 50.0, 100.0],
                                                                      ['hinge', 'squared_hinge'],
                                                                      ['l1', 'l2'],
                                                                      [True, False],
                                                                      [1e-1, 1e-2, 1e-3, 1e-4, 1e-5],
                                                                      [True, False]):
    features = input_data.drop('class', axis=1).values.astype(float)
    labels = input_data['class'].values

    try:
        # Create the pipeline for the model
        clf = make_pipeline(StandardScaler(),
                            LinearSVC(C=C,
                                      loss=loss,
                                      penalty=penalty,
                                      dual=dual,
                                      tol=tol,
                                      fit_intercept=fit_intercept))
        # 10-fold CV scores for the pipeline
        cv_scores = cross_val_score(estimator=clf, X=features, y=labels, cv=10)
    except KeyboardInterrupt:
        sys.exit(1)
    except:
        continue

    param_string = ''
    param_string += 'C={},'.format(C)
    param_string += 'loss={},'.format(loss)
    param_string += 'penalty={},'.format(penalty)
    param_string += 'dual={},'.format(dual)
    param_string += 'tol={},'.format(tol)
    param_string += 'fit_intercept={}'.format(fit_intercept)

    for cv_score in cv_scores:
        out_text = '\t'.join([dataset.split('/')[-1][:-7],
                              'LinearSVC',
                              param_string,
                              str(cv_score)])

        print(out_text)
        sys.stdout.flush()
