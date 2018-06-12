import os
import warnings

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

# Ignore warnings
warnings.filterwarnings("ignore")

plt.ion()   # interactive mode


class WindTurbinesDataset(Dataset):
    """Wind turbines data set."""

    def __init__(self, csv_file, root_dir, transform=None, subset='train'):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.frame = pd.read_csv(csv_file)
        self.frame = self.frame[self.frame['subset'] == subset]
        self.root_dir = root_dir
        self.transform = transform
        self.subset = subset
        self.target_map = {
            'positive': 1,
            'negative': 0,
        }

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir,
                                self.frame.image_file.values[idx])
        image = Image.open(img_name)
        # image = np.rollaxis(image, 0, 3)  # convert to channels first
        target = self.frame.contains_wind_turbine.values[idx]

        if self.transform:
            image = self.transform(image)

        return image, self.target_map[target]
