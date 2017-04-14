import sys
import pandas as pd
import numpy as np
import itertools
import warnings

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score, f1_score
from tpot_metrics import balanced_accuracy_score
from sklearn.pipeline import make_pipeline
import itertools

dataset = sys.argv[1]

# Read the data set into memory
input_data = pd.read_csv(dataset, compression='gzip', sep='\t').sample(frac=1., replace=False, random_state=42)

with warnings.catch_warnings():
    warnings.simplefilter('ignore')

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
        param_string += 'loss={},'.format(loss)
        param_string += 'fit_intercept={}'.format(fit_intercept)

        out_text = '\t'.join([dataset.split('/')[-1][:-7],
                              'PassiveAggressiveClassifier',
                              param_string,
                              str(accuracy),
                              str(macro_f1),
                              str(balanced_accuracy)])

        print(out_text)
        sys.stdout.flush()
