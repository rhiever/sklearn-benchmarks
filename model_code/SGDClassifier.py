import sys
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
import itertools

dataset = sys.argv[1]

# Read the data set into memory
input_data = pd.read_csv(dataset, compression='gzip', sep='\t')

for (loss, penalty, alpha, learning_rate, fit_intercept, l1_ratio, eta0, power_t) in itertools.product(['hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron'],
                                                                                                                ['l2', 'l1', 'elasticnet'],
                                                                                                                [0.000001, 0.00001, 0.0001, 0.001, 0.01],
                                                                                                                ['constant', 'optimal', 'invscaling'],
                                                                                                                [True, False],
                                                                                                                [0.0, 0.1, 0.15, 0.25, 0.5, 0.75, 0.9, 1.0],
                                                                                                                [0.0, 0.1, 0.5, 1.0, 10.0, 50.0, 100.0],
                                                                                                                [0.0, 0.1, 0.5, 1.0, 10.0, 50.0, 100.0]):
    if penalty != 'elasticnet' and l1_ratio != 0.15:
        continue

    if learning rate not in ['constant', 'invscaling'] and eta0 != 0.0:
        continue

    if learning_rate != 'invscaling' and power_t != 0.5:
        continue

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
            clf = SGDClassifier(loss=loss, penalty=penalty, alpha=alpha, learning_rate=learning_rate,
                                fit_intercept=fit_intercept, l1_ratio=l1_ratio, eta0=eta0, power_t=power_t)
            clf.fit(training_features, training_classes)
            testing_score = clf.score(testing_features, testing_classes)
        except KeyboardInterrupt:
            sys.exit(1)
        except:
            continue

        param_string = ''
        param_string += 'loss={},'.format(loss)
        param_string += 'penalty={},'.format(penalty)
        param_string += 'alpha={},'.format(alpha)
        param_string += 'learning_rate={},'.format(learning_rate)
        param_string += 'fit_intercept={},'.format(fit_intercept)
        param_string += 'l1_ratio={},'.format(l1_ratio)
        param_string += 'eta0={},'.format(eta0)
        param_string += 'power_t={}'.format(power_t)
    
        out_text = '\t'.join([dataset.split('/')[-1][:-7],
                              'SGDClassifier',
                              param_string,
                              str(testing_score)])

        print(out_text)
