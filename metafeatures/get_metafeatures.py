"""
Script to get metafeatures for all datasets stored in the ../data folder.
Dumps all the results in a csv called data_metafeatures.csv
"""

from glob import glob
import pandas as pd

from dataset_describe import Dataset
from collections import OrderedDict


def get_metafeatures(df):
    dataset = Dataset(df, dependent_col = 'class', prediction_type='classification')
   
    meta_features = OrderedDict()
    for i in dir(dataset):
        result = getattr(dataset, i)
        # print i
        if not i.startswith('__') and not i.startswith('_') and hasattr(result, '__call__'):
            meta_features[i] = result()

    return meta_features


def main():
    meta_features = []
    for i,dataset in enumerate(glob('../data/*')):
        # Read the data set into memory
        print 'Processing {0}'.format(dataset)
        input_data = pd.read_csv(dataset, compression='gzip', sep='\t')
        meta_features.append(get_metafeatures(input_data))
        
        # For testing purposes.
        # if i == 5:
        #     pd.DataFrame(meta_features).to_csv('data_metafeatures.csv')
        #     break

if __name__ == '__main__':
    main()