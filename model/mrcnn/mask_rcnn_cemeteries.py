import os

import numpy as np

import model as modellib
from config import Config
from cemeteries_dataset import CemeteriesDataset
import visualize


DATA_DIR = '/media/reinv/USB/cemeteries/'


class CemeteriesConfig(Config):
    """
    Derives from the base Config class and overrides values specific to the shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "cemeteries"
    IMAGE_RESIZE_MODE = "none"  # images have already been standardized to 1MP
    EPOCHS = 100

    # Image mean (RGB)
    MEAN_PIXEL = np.array([108.2, 118.6, 105.6])

    # These are 1MP images
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # background + windturbines

    # Since in general there is only one wind turbine per image
    RPN_TRAIN_ANCHORS_PER_IMAGE = 1

    # Maximum number of ground truth instances to use in one image
    MAX_GT_INSTANCES = 1

    # Max number of final detections
    DETECTION_MAX_INSTANCES = 1

    # Add custom image resolution property in meters per pixel
    RESOLUTION = 0.25


ROOT_DIR = os.path.abspath("../../")
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
config = CemeteriesConfig()

if __name__ == '__main__':
    config.display()

    model = modellib.MaskRCNN(mode="training", config=config,
                              model_dir=MODEL_DIR)

    # Training dataset
    print("Loading training set metadata:")
    dataset_train = CemeteriesDataset()
    dataset_train.load_samples(DATA_DIR, 'train', config)
    dataset_train.prepare()

    # Test on random images
    for _ in range(0):
        image_id = np.random.choice(dataset_train.image_ids)
        original_image, image_meta, gt_class_id, gt_bbox, gt_mask = \
            modellib.load_image_gt(dataset_train, config,
                                   image_id, use_mini_mask=False)

        print("image id", image_id)
        print("original_image", original_image)
        print("image_meta", image_meta)
        print("gt_class_id", gt_class_id)
        print("gt_bbox", gt_bbox)
        # print("gt_mask", gt_mask)
        metadata = dataset_train.image_info[image_id]
        print("geolocation", metadata['geolocation_rd'])

        visualize.display_instances(original_image, gt_bbox, gt_mask, gt_class_id,
                                    dataset_train.class_names, figsize=(8, 8))

    # Validation dataset
    print("Loading validation set metadata:")
    dataset_val = CemeteriesDataset()
    dataset_val.load_samples(DATA_DIR, 'validate', config)
    dataset_val.prepare()

    # Train the model
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=config.EPOCHS,
                layers='heads')
