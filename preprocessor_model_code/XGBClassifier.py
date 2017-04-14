import sys
import pandas as pd
import numpy as np
import itertools
import warnings

from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
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

    for (learning_rate, n_estimators, max_depth) in itertools.product([0.01, 0.1, 0.5, 1.0, 10.0, 50.0, 100.0],
                                                                      [10, 50, 100, 250],
                                                                      [1, 2, 3, 4, 5, 10, 20, 50, None]):
        features = input_data.drop('class', axis=1).values.astype(float)
        labels = input_data['class'].values

        try:
            # Create the pipeline for the model
            clf = make_pipeline(StandardScaler(),
                                XGBClassifier(learning_rate=learning_rate,
                                              n_estimators=n_estimators,
                                              max_depth=max_depth,
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
        param_string += 'learning_rate={},'.format(learning_rate)
        param_string += 'n_estimators={},'.format(n_estimators)
        param_string += 'max_depth={}'.format(max_depth)

        out_text = '\t'.join([dataset.split('/')[-1][:-7],
                              'XGBClassifier',
                              param_string,
                              str(accuracy),
                              str(macro_f1),
                              str(balanced_accuracy)])

        print(out_text)
        sys.stdout.flush()
