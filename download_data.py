'''
This script downloads all of the data for the benchmark to the 'data' directory.
You will need an internet connection to run this script.
'''
from __future__ import print_function
import os

print('Making sure that you have the necessary Python libraries installed...\n')

try:
    import wget
except:
    try:
        import pip
    except:
        os.system('easy_install pip')
    
    os.system('pip install wget')
    print('')

import wget
print('Downloading the data sets...')

if not os.path.isdir('data'):
    os.mkdir('data')

dataset_list = ['20newsgroups.csv.gz',
                'GAMETES-easy-4x2-way_her-0.4_pop-1600_attribs-100_discrete.csv.gz',
                'GAMETES-hard-4x2-way_her-0.1_pop-200_attribs-100_discrete.csv.gz',
                'Hill_Valley_with_noise.csv.gz',
                'Hill_Valley_without_noise.csv.gz',
                'breast-cancer-wisconsin.csv.gz',
                'car-evaluation.csv.gz',
                'cifar-10.csv.gz',
                'cifar-100-coarse.csv.gz',
                'cifar-100-fine.csv.gz',
                'ecoli.csv.gz',
                'flags.csv.gz',
                'glass.csv.gz',
                'ionosphere.csv.gz',
                'mnist.csv.gz',
                'sat.csv.gz',
                'spambase.csv.gz',
                'svhn.csv.gz',
                'wine-quality-red.csv.gz',
                'wine-quality-white.csv.gz',
                'wine-recognition.csv.gz',
                'yeast.csv.gz']

try:
    for dataset in dataset_list:
        if not os.path.exists('data/{}'.format(dataset)):
            url = 'http://www.randalolson.com/data/{}'.format(dataset)
            print('\n\n' + url)
            wget.download(url, out='data/')

    print('')
    
except KeyboardInterrupt:
    os.system('rm *.csv.gz*.tmp')
    print('')
