import sys
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
import itertools

dataset = sys.argv[1]

# Read the data set into memory
input_data = pd.read_csv(dataset, compression='gzip', sep='\t')

for (max_depth, max_features, criterion) in itertools.product([1, 2, 3, 4, 5, 10, 20, 50, None],
                                                              [0.1, 0.25, 0.5, 0.75, 'sqrt', 'log2', None],
                                                              ['gini', 'entropy']):
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
            clf = DecisionTreeClassifier(max_depth=max_depth, max_features=max_features, criterion=criterion)
            clf.fit(training_features, training_classes)
            testing_score = clf.score(testing_features, testing_classes)
        except KeyboardInterrupt:
            sys.exit(1)
        except:
            continue
    
        param_string = ''
        param_string += 'max_depth={},'.format(max_depth)
        param_string += 'max_features={},'.format(max_features)
        param_string += 'criterion={}'.format(criterion)
    
        out_text = '\t'.join([dataset.split('/')[-1][:-7],
                              'DecisionTreeClassifier',
                              param_string,
                              str(testing_score)])
    
        print(out_text)
