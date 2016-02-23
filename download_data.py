'''
This script downloads all of the data for the benchmark to the 'data' directory.
You will need an internet connection to run this script.
'''
from __future__ import print_function
import os
import requests
from bs4 import BeautifulSoup
import wget

print('Downloading the data sets...')

if not os.path.isdir('data'):
    os.mkdir('data')

dataset_list = []
soup = BeautifulSoup(requests.get('http://www.randalolson.com/data/benchmarks/').text, 'lxml')

for a in soup.find_all('a'):
    data_file = a.text.strip()
    if '.csv.gz' not in data_file:
        continue
    dataset_list.append(data_file)

try:
    for dataset in dataset_list:
        dataset_name = dataset.split('/')[-1]
        if not os.path.exists('data/{}'.format(dataset_name)):
            url = 'http://www.randalolson.com/data/benchmarks/{}'.format(dataset)
            print('\n\n' + url)
            wget.download(url, out='data/')

    print('')

except KeyboardInterrupt:
    os.system('rm *.csv.gz*.tmp')
    print('')
