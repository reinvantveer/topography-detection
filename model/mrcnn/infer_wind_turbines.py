import os

import matplotlib.pyplot as plt
import numpy as np
import argparse

from mask_rcnn_wind_turbines import WindTurbinesConfig, DATA_DIR
import model as modellib
from wind_turbines_dataset import WindTurbinesDataset
import visualize

parser = argparse.ArgumentParser(description='Infer random a validation image from a trained mask-rcnn model.')
parser.add_argument('model_weights', metavar='model_weights', type=str, help='a keras h5 mask-rcnn model weights file')
args = parser.parse_args()

MODEL_DIR = os.path.dirname(args.model_weights)


class InferenceConfig(WindTurbinesConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


def get_ax(rows=1, cols=1, size=8):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.

    Change the default size attribute to control the size
    of rendered images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size * cols, size * rows))
    return ax


inference_config = InferenceConfig()

# Validation dataset
print("Loading validation set metadata:")
dataset_val = WindTurbinesDataset()
dataset_val.load_samples(DATA_DIR, 'validate', inference_config)
dataset_val.prepare()

# Recreate the model in inference mode
model = modellib.MaskRCNN(mode="inference",
                          config=inference_config,
                          model_dir=MODEL_DIR)

# Get path to saved weights
# Either set a specific path or find last trained weights
# model_path = os.path.join(ROOT_DIR, ".h5 file name here")
# model_path = model.find_last()
model_path = args.model_weights

# Load trained weights
print("Loading weights from ", model_path)
model.load_weights(model_path, by_name=True)

# Test on random images
for _ in range(10):
    image_id = np.random.choice(dataset_val.image_ids)
    original_image, image_meta, gt_class_id, gt_bbox, gt_mask = \
        modellib.load_image_gt(dataset_val, inference_config,
                               image_id, use_mini_mask=False)

    print("image id", image_id)
    print("original_image", original_image)
    print("image_meta", image_meta)
    print("gt_class_id", gt_class_id)
    print("gt_bbox", gt_bbox)
    # print("gt_mask", gt_mask)
    metadata = dataset_val.image_info[image_id]
    print("geolocation", metadata['geolocation_rd'])

    visualize.display_instances(original_image, gt_bbox, gt_mask, gt_class_id,
                                dataset_val.class_names, figsize=(8, 8))

    results = model.detect([original_image], verbose=1)

    r = results[0]
    visualize.display_instances(original_image, r['rois'], r['masks'], r['class_ids'],
                                dataset_val.class_names, r['scores'], ax=get_ax())
print('Done!')
