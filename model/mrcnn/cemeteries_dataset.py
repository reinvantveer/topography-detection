import os

import pandas as pd
import numpy as np
import skimage
from shapely import wkt
from tqdm import tqdm

import utils


class CemeteriesDataset(utils.Dataset):

    def load_samples(self, dataset_dir, subset, config):
        """Return the requested number of images.
        count: number of images to generate.
        height, width: the size of the generated images.
        """
        assert subset in ["train", "validate"]
        CSV_FILE = dataset_dir + 'metadata.csv'

        frame = pd.read_csv(CSV_FILE)
        frame = frame[frame['subset'] == subset]
        frame = frame[frame['contains'] == 'positive']

        # Add classes
        self.add_class("cemeteries", 1, "cemetery")

        # Add images
        # Images are loaded in load_image().
        rows = list(frame.iterrows())
        for _, record in tqdm(rows):
            offset_x = float(record['offset_x']) / config.RESOLUTION / record['scale']
            offset_y = float(record['offset_y']) / config.RESOLUTION / record['scale']
            centroid_pixel_x = config.IMAGE_SHAPE[0] / 2 - offset_x
            centroid_pixel_y = config.IMAGE_SHAPE[1] / 2 + offset_y

            if record['image_file'].startswith(dataset_dir):
                image_file = record['image_file']
            else:
                image_file = dataset_dir + record['image_file']

            if os.path.isfile(image_file):
                self.add_image("cemeteries",
                               image_id=record['image_file'], path=image_file,
                               width=config.IMAGE_SHAPE[0], height=config.IMAGE_SHAPE[1],
                               mask_shape=record['mask_shape'],
                               geolocation_rd=(float(record['original_rd_x']), float(record['original_rd_y'])),
                               geolocation_offset=(float(record['offset_x']), float(record['offset_y'])),
                               scale=float(record['scale']), resolution=config.RESOLUTION,
                               )
            else:
                print('Skipping non-existent path', image_file)

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        image_info = self.image_info[image_id]
        if image_info["source"] != "cemeteries":
            return super(self.__class__, self).load_mask(image_id)

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        mask_shape = wkt.loads(info['mask_shape'])

        if mask_shape.geom_type == 'Polygon':
            mask = np.zeros([info["height"], info["width"], 1],
                        dtype=np.uint8)
            # Get indexes of pixels inside the polygon and set them to 1
            x_coords = np.array(mask_shape.boundary.coords.xy[0]) - info['geolocation_rd'][0]
            x_coords = (x_coords - info['geolocation_offset'][0]) / info['resolution'] / info['scale']
            x_coords = x_coords + info['width'] / 2
            y_coords = np.array(mask_shape.boundary.coords.xy[1]) - info['geolocation_rd'][1]
            y_coords = (-y_coords + info['geolocation_offset'][1]) / info['resolution'] / info['scale']
            y_coords = y_coords + info['height'] / 2

            rr, cc = skimage.draw.polygon(y_coords, x_coords, mask.shape)
            mask[rr, cc, 0] = 1
        else:
            raise ValueError('Don\'t know how to handle geometries of type {} for record {}'.format(mask_shape.geom_type, info['path']))

        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s

        return mask.astype(np.bool), np.ones([mask.shape[-1]], dtype=np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "cemeteries":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)