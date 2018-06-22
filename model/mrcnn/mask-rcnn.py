import os


import mrcnn.model as modellib
from mrcnn.config import Config
from mrcnn.random_shape_dataset import ShapesDataset
from mrcnn.wind_turbines_dataset import WindTurbinesDataset

DATA_DIR = '/media/reinv/501E7A121E79F0F8/data/windturbines/'


class WindTurbinesConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "windturbines"
    IMAGE_RESIZE_MODE = "none"  # images have already been standardized to 1MP

    # These are 1MP images
    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # background + windturbines

    # Since in general there is only one wind turbine per image
    RPN_TRAIN_ANCHORS_PER_IMAGE = 1

    # Maximum number of ground truth instances to use in one image
    MAX_GT_INSTANCES = 10

    # Max number of final detections
    DETECTION_MAX_INSTANCES = 3

    # Add custom image resolution property in meters per pixel
    RESOLUTION = 0.50

    # Add custom object size property as diameter in pixels
    OBJECT_SIZE = 20


ROOT_DIR = os.path.abspath("../../")
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
config = WindTurbinesConfig()
config.display()


model = modellib.MaskRCNN(mode="training", config=config,
                          model_dir=MODEL_DIR)

# Training dataset
# dataset_train = ShapesDataset()
dataset_train = WindTurbinesDataset()
dataset_train.load_samples(DATA_DIR, 'train', config)
dataset_train.prepare()

# Validation dataset
# dataset_val = ShapesDataset()
dataset_val = WindTurbinesDataset()
dataset_val.load_samples(DATA_DIR, 'validate', config)
dataset_val.prepare()

model.train(dataset_train, dataset_val,
            learning_rate=config.LEARNING_RATE,
            epochs=1,
            layers='heads')