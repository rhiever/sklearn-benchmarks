import sys
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler

dataset = sys.argv[1]

# Read the data set into memory
input_data = pd.read_csv(dataset, compression='gzip', sep='\t')


for dataset_repeat in range(1, 31):
    # Divide the data set into a training and testing sets, each time with a different RNG seed
    training_indices, testing_indices = next(iter(StratifiedShuffleSplit(input_data['class'].values,
                                                                         n_iter=1,
                                                                         train_size=0.75,
                                                                         test_size=0.25,
                                                                         random_state=dataset_repeat)))

    training_features = input_data.loc[training_indices].drop('class', axis=1).values
    training_classes = input_data.loc[training_indices, 'class'].values

    testing_features = input_data.loc[testing_indices].drop('class', axis=1).values
    testing_classes = input_data.loc[testing_indices, 'class'].values

    ss = StandardScaler()
    training_features = ss.fit_transform(training_features.astype(float))
    testing_features = ss.transform(testing_features.astype(float))

    # Create and fit the model on the training data
    try:
        clf = GaussianNB()
        clf.fit(training_features, training_classes)
        testing_score = clf.score(testing_features, testing_classes)
    except KeyboardInterrupt:
        sys.exit(1)
    except:
        continue

    param_string = ''

    out_text = '\t'.join([dataset.split('/')[-1][:-7],
                          'GaussianNB',
                          param_string,
                          str(testing_score)])

    print(out_text)
