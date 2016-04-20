import sys
import pandas as pd
import itertools
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.cross_validation import cross_val_score
from sklearn.pipeline import make_pipeline
import itertools

dataset = sys.argv[1]

# Read the data set into memory
input_data = pd.read_csv(dataset, compression='gzip', sep='\t')

for (C, gamma, kernel, degree, coef0, tol) in itertools.product([0.01, 0.1, 0.5, 1.0, 10.0, 50.0, 100.0],
                                                                [0.01, 0.1, 0.5, 1.0, 10.0, 50.0, 100.0, 'auto'],
                                                                ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed'],
                                                                [2, 3],
                                                                [0.0, 0.1, 0.5, 1.0, 10.0, 50.0, 100.0],
                                                                [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]):
    if kernel != 'poly' and degree > 2:
        continue

    if kernel not in ['rbf', 'poly', 'sigmoid'] and gamma != 'auto':
        continue

    if kernel not in ['poly', 'sigmoid'] and coef0 != 0.0:
        continue

    features = input_data.drop('class', axis=1).values.astype(float)
    labels = input_data['class'].values

    try:
        # Create the pipeline for the model
        clf = make_pipeline(StandardScaler(),
                            SVC(C=C,
                                gamma=gamma,
                                kernel=kernel,
                                degree=degree,
                                coef0=coef0,
                                tol=tol))
    
        # 10-fold CV scores for the pipeline
        cv_scores = cross_val_score(estimator=clf, X=features, y=labels, cv=10)
    except KeyboardInterrupt:
        sys.exit(1)
    except:
        continue

    param_string = ''
    param_string += 'C={},'.format(C)
    param_string += 'gamma={},'.format(gamma)
    param_string += 'kernel={},'.format(kernel)
    param_string += 'degree={},'.format(degree)
    param_string += 'coef0={},'.format(coef0)
    param_string += 'tol={}'.format(tol)

    for cv_score in cv_scores:
        out_text = '\t'.join([dataset.split('/')[-1][:-7],
                              'SVC',
                              param_string,
                              str(cv_score)])

        print(out_text)
        sys.stdout.flush()
