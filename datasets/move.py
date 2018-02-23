import csv

import os

from os.path import isfile
from shutil import move

DATA_DIR = '../data/windturbines/'

reader = csv.DictReader(open('_metadata.csv'))
meta = open('_metadata2.csv', 'w')
writer = csv.DictWriter(meta, fieldnames=reader.fieldnames)
writer.writeheader()

# Data dirs
for subset in ['train', 'validate', 'test']:
    os.makedirs(DATA_DIR + subset + '/positive', exist_ok=True)
    os.makedirs(DATA_DIR + subset + '/negative', exist_ok=True)

for index, row in enumerate(reader):
    row_copy = dict.copy(row)
    sub_idx = int(index / 5)

    subset = row_copy['subset']
    if sub_idx % 10 == 0:
        row_copy['subset'] = 'validate'

    row_copy['image_file'] = row_copy['image_file'].replace(
        subset, row_copy['subset'] + '/' + row_copy['contains_wind_turbine'])

    if isfile(row['image_file']):
        move(row['image_file'], row_copy['image_file'])
    else:
        raise ValueError('unable to find {}'.format(row['image_file']))

    writer.writerow(row_copy)
    meta.flush()
