import sys
import pandas as pd
import numpy as np
import itertools
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.cross_validation import cross_val_score
from sklearn.pipeline import make_pipeline
import itertools

dataset = sys.argv[1]

# Read the data set into memory
input_data = pd.read_csv(dataset, compression='gzip', sep='\t')

features = input_data.drop('class', axis=1).values.astype(float)
labels = input_data['class'].values

try:
    # Create the pipeline for the model
    clf = make_pipeline(StandardScaler(),
                        GaussianNB())
    # 10-fold CV scores for the pipeline with a fixed seed
    np.random.seed(2097483)
except KeyboardInterrupt:
    sys.exit(1)
except:
    sys.exit(0)

param_string = ''

for cv_score in cv_scores:
    out_text = '\t'.join([dataset.split('/')[-1][:-7],
                          'GaussianNB',
                          param_string,
                          str(cv_score)])

    print(out_text)
    sys.stdout.flush()
