import csv
import json
import os
import re
from datetime import datetime, timedelta
from os.path import isfile
from time import time
from urllib.parse import quote
import requests
from numpy.random import random

# Some metadata first

SCRIPT_VERSION = '0.0.3'
SCRIPT_NAME = os.path.basename(__file__)
TIMESTAMP = str(datetime.now()).replace(':', '.')
SCRIPT_START = time()

BRT_OBJECT_TYPE = 'Windturbine'
MAX_SCALE_FACTOR = 0.1
MAX_COORD_OFFSET = 100
IMAGE_SIZE_X = 1024  # in pixels
IMAGE_SIZE_Y = 1024  # in pixels
RESOLUTION = 0.25    # in meters per pixel along x and y axis
BBOX_CENTER_OFFSET_X = IMAGE_SIZE_X * RESOLUTION / 2
BBOX_CENTER_OFFSET_Y = IMAGE_SIZE_Y * RESOLUTION / 2
IMAGES_PER_OBJECT = 5
TRAIN_TEST_SPLIT = 1/10
DATA_DIR = '../data/windturbines/'
RATIO_POS_NEG_SAMPLES = 1/10
RD_X_MIN = 646.36
RD_X_MAX = 308975.28
RD_Y_MIN = 276050.82
RD_Y_MAX = 636456.31

url = "https://data.pdok.nl/sparql"
payload = '''
PREFIX brt: <http://brt.basisregistraties.overheid.nl/def/top10nl#>
PREFIX geo: <http://www.opengis.net/ont/geosparql#>
prefix pdok: <http://data.pdok.nl/def/pdok#>

SELECT  * WHERE {{
 ?instance a ?class ;
    geo:hasGeometry/pdok:asWKT-RD ?rd ;
    geo:hasGeometry/geo:asWKT ?wgs84 .
  filter(
    ?class = brt:{} 
  )
}}
'''.format(BRT_OBJECT_TYPE)

headers = {
    'Accept': "application/sparql-results+json",
    'Content-Type': "application/x-www-form-urlencoded; charset=UTF-8",
    'X-Requested-With': "XMLHttpRequest",
    'Connection': "keep-alive",
}

response = requests.request("POST", url, data='query=' + quote(payload), headers=headers)
if not response.status_code == 200:
    print('Error getting list of', BRT_OBJECT_TYPE, 'instances from sparql endpoint')
response_dict = json.loads(response.text)
variables = response_dict['head']['vars']
positive_data_points = response_dict['results']['bindings']

os.makedirs(DATA_DIR + 'train', exist_ok=True)
os.makedirs(DATA_DIR + 'test', exist_ok=True)

# Metadata csv creation/append
train_csv_exists = isfile('{}train/metadata.csv'.format(DATA_DIR))
test_csv_exists = isfile('{}test/metadata.csv'.format(DATA_DIR))
csvfile = {
    'train': open('{}train/metadata.csv'.format(DATA_DIR), 'a', newline=''),
    'test': open('{}test/metadata.csv'.format(DATA_DIR), 'a', newline='')
}
fieldnames = ['timestamp', 'contains_wind_turbine', 'URI', 'image_file', 'original_rd_x', 'original_rd_y', 'offset_x',
              'offset_y', 'scale', 'request', ]
csv_writer = {
    'train': csv.DictWriter(csvfile['train'], fieldnames),
    'test': csv.DictWriter(csvfile['test'], fieldnames)
}

if not train_csv_exists:
    csv_writer['train'].writeheader()
if not test_csv_exists:
    csv_writer['test'].writeheader()

# Harvest images for locations with wind turbines
for record_index, record in enumerate(positive_data_points):
    if record_index % (1 / TRAIN_TEST_SPLIT) == 0:
        subset = 'test'
    else:
        subset = 'train'

    random_offsets = 2 * MAX_COORD_OFFSET * (random((IMAGES_PER_OBJECT, 2)) - 0.5)
    random_scales = 1 - 2 * MAX_SCALE_FACTOR * (random((IMAGES_PER_OBJECT,)) - 0.5)
    uri = record['instance']['value']
    brt_id = re.findall('.*/(.+)', uri)[0]

    for image_index, (offset, scale) in enumerate(zip(random_offsets, random_scales)):
        image_file_name = brt_id + '-' + str(image_index) + '.png'
        if isfile(DATA_DIR + 'train/' + image_file_name) or isfile(DATA_DIR + 'test/' + image_file_name):
            print('Already have', image_file_name)
            continue

        rd_wkt = record['rd']['value']
        rd_coords = re.findall('POINT \((.+) (.+)\)', rd_wkt)
        if not rd_coords:
            print('Error finding coordinates in RD geometry string', rd_wkt)
            continue

        rd_x = float(rd_coords[0][0]) + offset[0]
        rd_y = float(rd_coords[0][1]) + offset[1]

        url = "https://geodata.nationaalgeoregister.nl/luchtfoto/rgb/wms"
        querystring = {
            "LAYERS": "2016_ortho25",
            "FORMAT": "image/png",
            "TRANSPARENT": "TRUE",
            "SERVICE": "WMS",
            "VERSION": "1.1.1",
            "REQUEST": "GetMap",
            "STYLES": "",
            "SRS": "EPSG:28992",
            # at scale in meter-based coordinate systems it is useless to have more than one decimal
            "BBOX": "{min_x:0.1f},{min_y:0.1f},{max_x:0.1f},{max_y:0.1f}".format(
                min_x=rd_x - (BBOX_CENTER_OFFSET_X * scale),
                min_y=rd_y - (BBOX_CENTER_OFFSET_Y * scale),
                max_x=rd_x + (BBOX_CENTER_OFFSET_X * scale),
                max_y=rd_y + (BBOX_CENTER_OFFSET_Y * scale),
            ),
            "WIDTH": IMAGE_SIZE_X, "HEIGHT": IMAGE_SIZE_Y
        }

        image_file_path = DATA_DIR + subset + '/' + image_file_name
        response = requests.request("GET", url, params=querystring)
        with open(image_file_path, mode='wb') as image:
            for chunk in response:
                image.write(chunk)

        csv_writer[subset].writerow({
            'timestamp': datetime.now(),
            'contains_wind_turbine': True,
            'URI': uri,
            'image_file': image_file_path,
            'original_rd_x': rd_coords[0][0],
            'original_rd_y': rd_coords[0][1],
            'offset_x': offset[0],
            'offset_y': offset[1],
            'scale': scale,
            'request': response.request.url
        })
        csvfile[subset].flush()
        print('Wrote image and data for record', record_index, 'of', len(positive_data_points))

number_of_neg_samples = int(len(positive_data_points) / RATIO_POS_NEG_SAMPLES)

for neg_sample_index in range(number_of_neg_samples):
    if neg_sample_index % (1 / TRAIN_TEST_SPLIT) == 0:
        subset = 'test'
    else:
        subset = 'train'

    random_x = random((1,)) * (RD_X_MAX - RD_X_MIN) + RD_X_MIN
    random_y = random((1,)) * (RD_Y_MAX - RD_Y_MIN) + RD_Y_MIN
    random_offsets = 2 * MAX_COORD_OFFSET * (random((IMAGES_PER_OBJECT, 2)) - 0.5)
    random_scales = 1 - 2 * MAX_SCALE_FACTOR * (random((IMAGES_PER_OBJECT,)) - 0.5)

    for image_index, (offset, scale) in enumerate(zip(random_offsets, random_scales)):
        image_file_name = 'negative-' + str(neg_sample_index) + '-' + str(image_index) + '.png'
        if isfile(DATA_DIR + 'train/' + image_file_name) or isfile(DATA_DIR + 'test/' + image_file_name):
            print('Already have', image_file_name)
            continue

        rd_x = float(random_x) + offset[0]
        rd_y = float(random_y) + offset[1]
        url = "https://geodata.nationaalgeoregister.nl/luchtfoto/rgb/wms"
        querystring = {
            "LAYERS": "2016_ortho25",
            "FORMAT": "image/png",
            "TRANSPARENT": "TRUE",
            "SERVICE": "WMS",
            "VERSION": "1.1.1",
            "REQUEST": "GetMap",
            "STYLES": "",
            "SRS": "EPSG:28992",
            # at scale in meter-based coordinate systems it is useless to have more than one decimal
            "BBOX": "{min_x:0.1f},{min_y:0.1f},{max_x:0.1f},{max_y:0.1f}".format(
                min_x=rd_x - (BBOX_CENTER_OFFSET_X * scale),
                min_y=rd_y - (BBOX_CENTER_OFFSET_Y * scale),
                max_x=rd_x + (BBOX_CENTER_OFFSET_X * scale),
                max_y=rd_y + (BBOX_CENTER_OFFSET_Y * scale),
            ),
            "WIDTH": IMAGE_SIZE_X, "HEIGHT": IMAGE_SIZE_Y
        }

        image_file_path = DATA_DIR + subset + '/' + image_file_name
        response = requests.request("GET", url, params=querystring)
        with open(image_file_path, mode='wb') as image:
            for chunk in response:
                image.write(chunk)

        csv_writer[subset].writerow({
            'timestamp': datetime.now(),
            'contains_wind_turbine': False,
            'URI': None,
            'image_file': image_file_path,
            'original_rd_x': random_x,
            'original_rd_y': random_y,
            'offset_x': offset[0],
            'offset_y': offset[1],
            'scale': scale,
            'request': response.request.url
        })
        csvfile[subset].flush()
        print('Wrote image and data for record', neg_sample_index, 'of', number_of_neg_samples)

runtime = time() - SCRIPT_START
print(SCRIPT_NAME, 'finished successfully in {}'.format(timedelta(seconds=runtime)))
