import pandas as pd
import numpy as np
import skimage
from tqdm import tqdm

import utils


class WindTurbinesDataset(utils.Dataset):

    def load_samples(self, dataset_dir, subset, config):
        """Return the requested number of images.
        count: number of images to generate.
        height, width: the size of the generated images.
        """
        assert subset in ["train", "validate"]
        CSV_FILE = dataset_dir + 'metadata.csv'

        frame = pd.read_csv(CSV_FILE)
        frame = frame[frame['subset'] == subset]
        frame = frame[frame['contains_wind_turbine'] == 'positive']

        # Add classes
        self.add_class("windturbines", 1, "windturbine")

        # Add images
        # Images are loaded in load_image().
        rows = list(frame.iterrows())
        for _, record in tqdm(rows):
            offset_x = float(record['offset_x']) / config.RESOLUTION * record['scale']
            offset_y = float(record['offset_y']) / config.RESOLUTION * record['scale']
            centroid_pixel_x = config.IMAGE_SHAPE[0] / 2 - offset_x
            centroid_pixel_y = config.IMAGE_SHAPE[1] / 2 + offset_y

            self.add_image("windturbines",
                           image_id=record['image_file'], path=dataset_dir + record['image_file'],
                           width=config.IMAGE_SHAPE[0], height=config.IMAGE_SHAPE[1],
                           centroids=[(round(centroid_pixel_y), round(centroid_pixel_x))],  # kinda weird HxW
                           size=config.OBJECT_SIZE / 2)

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a balloon dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "windturbines":
            return super(self.__class__, self).load_mask(image_id)

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        mask = np.zeros([info["height"], info["width"], len(info["centroids"])],  # prepared for multiple instances
                        dtype=np.uint8)
        for i, centroid in enumerate(info["centroids"]):
            # Get indexes of pixels inside the polygon and set them to 1
            rr, cc = skimage.draw.circle(centroid[0], centroid[1], 10)
            mask[rr, cc, i] = 1

        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s

        return mask.astype(np.bool), np.ones([mask.shape[-1]], dtype=np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "windturbines":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)