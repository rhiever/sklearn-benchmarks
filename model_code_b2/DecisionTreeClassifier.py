import sys
import pandas as pd
import numpy as np
import itertools
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.pipeline import make_pipeline

dataset = sys.argv[1]

# Read the data set into memory
input_data = pd.read_csv(dataset, compression='gzip', sep='\t')

for (max_depth, max_features,
     criterion, min_weight_fraction_leaf) in itertools.product([None],
                                                               [None],
                                                               ['gini', 'entropy'],
                                                               np.arange(0., 0.51, 0.05)):
    features = input_data.drop('class', axis=1).values.astype(float)
    labels = input_data['class'].values

    try:
        # Create the pipeline for the model
        clf = make_pipeline(StandardScaler(),
                            DecisionTreeClassifier(max_depth=max_depth,
                                                   max_features=max_features,
                                                   criterion=criterion,
                                                   min_weight_fraction_leaf=min_weight_fraction_leaf))
        # 10-fold CV scores for the pipeline
        cv_scores = cross_val_score(estimator=clf, X=features, y=labels, cv=10)
    except KeyboardInterrupt:
        sys.exit(1)
    except:
        continue

    param_string = ''
    param_string += 'max_depth={},'.format(max_depth)
    param_string += 'max_features={},'.format(max_features)
    param_string += 'criterion={},'.format(criterion)
    param_string += 'min_weight_fraction_leaf={}'.format(min_weight_fraction_leaf)

    for cv_score in cv_scores:
        out_text = '\t'.join([dataset.split('/')[-1][:-7],
                              'DecisionTreeClassifier',
                              param_string,
                              str(cv_score)])

        print(out_text)
        sys.stdout.flush()
