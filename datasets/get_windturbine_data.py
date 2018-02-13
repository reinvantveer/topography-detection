import os
from uuid import uuid4
from time import time
from datetime import datetime, timedelta
import json
import re
from urllib.parse import quote

import numpy as np
import requests
import csv

# Some metadata first
SCRIPT_VERSION = '0.0.1'
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
    'Host': "data.pdok.nl",
    'User-Agent': "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:58.0) Gecko/20100101 Firefox/58.0",
    'Accept': "application/sparql-results+json",
    'Accept-Language': "en-US,en;q=0.5",
    'Referer': "https://data.pdok.nl/sparql",
    'Content-Type': "application/x-www-form-urlencoded; charset=UTF-8",
    'X-Requested-With': "XMLHttpRequest",
    'Cookie': "_ga=GA1.2.1871540942.1516202440",
    'Connection': "keep-alive",
    'Cache-Control': "no-cache",
    'Postman-Token': "357b09ee-be3c-a7e4-db09-cf02c871603d"
}

response = requests.request("POST", url, data='query=' + quote(payload), headers=headers)
response_dict = json.loads(response.text)
vars = response_dict['head']['vars']
data_points = response_dict['results']['bindings']
random_offsets = 2 * MAX_COORD_OFFSET * (np.random.random((len(data_points), 2)) - 0.5)
random_scales = 1 - 2 * MAX_SCALE_FACTOR * (np.random.random((len(data_points),)) - 0.5)

with open('../data/windturbines/metadata{}.csv'.format(TIMESTAMP), 'w', newline='') as csvfile:
    fieldnames = ['timestamp', 'URI', 'image_file', 'original_rd_x', 'original_rd_y', 'offset_x', 'offset_y', 'scale',
                  'request', ]
    csvwriter = csv.DictWriter(csvfile, fieldnames)

    for record in data_points:
        random_offsets = 2 * MAX_COORD_OFFSET * (np.random.random((IMAGES_PER_OBJECT, 2)) - 0.5)
        random_scales = 1 - 2 * MAX_SCALE_FACTOR * (np.random.random((IMAGES_PER_OBJECT,)) - 0.5)

        for offset, scale in zip(random_offsets, random_scales):
            rd_wkt = record['rd']['value']
            uri = record['instance']['value']
            rd_pattern = re.compile('POINT \((.+) (.+)\)')
            rd_coords = rd_pattern.findall(rd_wkt)
            if not rd_coords:
                print('Error finding coordinates in RD geometry string', rd_wkt)
                continue

            url = "https://geodata.nationaalgeoregister.nl/luchtfoto/rgb/wms"
            rd_x = float(rd_coords[0][0]) + offset[0]
            rd_y = float(rd_coords[0][1]) + offset[1]
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

            headers = {
                'Host': "geodata.nationaalgeoregister.nl",
                'User-Agent': "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:58.0) Gecko/20100101 Firefox/58.0",
                'Accept': "*/*",
                'Accept-Language': "en-US,en;q=0.5",
                'Referer': "http://pdokviewer.pdok.nl/",
                'Connection': "keep-alive",
                'Cache-Control': "no-cache",
                'Postman-Token': "c0f65097-d71e-4254-efd5-ee2b470c2cb8"
            }

            image_file_path = '../data/windturbines/' + str(uuid4()) + '.png'
            response = requests.request("GET", url, headers=headers, params=querystring)
            with open(image_file_path, mode='wb') as image:
                for chunk in response:
                    image.write(chunk)

            csvwriter.writerow({
                'timestamp': datetime.now(),
                'URI': uri,
                'image_file': image_file_path,
                'original_rd_x': rd_coords[0][0],
                'original_rd_y': rd_coords[0][1],
                'offset_x': offset[0],
                'offset_y': offset[1],
                'scale': scale,
                'request': response.request.url
            })
            csvfile.flush()

runtime = time() - SCRIPT_START
print(SCRIPT_NAME, 'finished successfully in {}'.format(timedelta(seconds=runtime)))
