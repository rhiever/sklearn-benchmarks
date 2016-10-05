import sys
import pandas as pd
import numpy as np
import itertools
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score, f1_score
from tpot_metrics import balanced_accuracy_score
from sklearn.pipeline import make_pipeline
import itertools

dataset = sys.argv[1]

# Read the data set into memory
input_data = pd.read_csv(dataset, compression='gzip', sep='\t').sample(frac=1., replace=False, random_state=42)

for (C, penalty, fit_intercept, dual) in itertools.product(np.arange(0.5, 20.1, 0.5),
                                                           ['l1', 'l2'],
                                                           [True, False],
                                                           [True, False]):
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
    param_string += 'C={},'.format(C)
    param_string += 'penalty={},'.format(penalty)
    param_string += 'fit_intercept={},'.format(fit_intercept)
    param_string += 'dual={},'.format(dual)

    out_text = '\t'.join([dataset.split('/')[-1][:-7],
                          'LogisticRegression',
                          param_string,
                          str(accuracy),
                          str(macro_f1),
                          str(balanced_accuracy)])

    print(out_text)
    sys.stdout.flush()
