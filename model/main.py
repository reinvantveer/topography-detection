import os
from datetime import datetime

import numpy as np
import torch
from tensorboard_logger import configure
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.transforms import transforms

from data_loader import WindTurbinesDataset
from recurrent_attention import RecurrentAttention
from train import train_one_epoch, validate

SCRIPT_VERSION = '2.0.5'
SCRIPT_NAME = 'pytorch RAM'
TIMESTAMP = str(datetime.now()).replace(':', '.')
SIGNATURE = SCRIPT_NAME + ' ' + SCRIPT_VERSION + ' ' + TIMESTAMP

# Add max pooling to reduce parameters and increase batch size?
# Add soft attention mechanism?
# Tweak REINFORCE gaussian policy standard deviation

hp = {
    # Data settings
    'ROOT_DIR': os.getenv('ROOT_DIR', '/media/reinv/501E7A121E79F0F8/data/windturbines/'),
    'PLOT_DIR': os.getenv('PLOT_DIR', '../plot/'),
    'NUM_CHANNELS': int(os.getenv('NUM_CHANNELS', 3)),          # channels per image
    'NUM_CLASSES': int(os.getenv('NUM_CLASSES', 2)),            # number of classes for data set

    # Training settings
    'EPOCHS': int(os.getenv('EPOCHS', 200)),
    'PATIENCE': int(os.getenv('PATIENCE', 50)),
    'LEARNING_RATE': float(os.getenv('LEARNING_RATE', 3e-4)),
    'BATCH_SIZE': int(os.getenv('BATCH_SIZE', 32)),
    'VALIDATE_BATCH_SIZE': int(os.getenv('VALIDATE_BATCH_SIZE', 16)),
    'PLOT_FREQ': int(os.getenv('PLOT_FREQ', 1)),
    'LOGS_DIR': os.getenv('LOGS_DIR', './tensorboard_log'),
    'MODEL_NAME': os.getenv('MODEL_NAME', SIGNATURE),

    # Glimpse network
    'PATCH_SIZE': int(os.getenv('PATCH_SIZE', 48)),             # size of extracted patch at highest res
    'NUM_PATCHES': int(os.getenv('NUM_PATCHES', 3)),            # number of downscaled patches per glimpse
    'GLIMPSE_SCALE': int(os.getenv('GLIMPSE_SCALE', 3)),        # scale of successive patches
    'LOC_HIDDEN': int(os.getenv('LOC_HIDDEN', 128)),            # hidden size of loc fully connected layer
    'GLIMPSE_HIDDEN': int(os.getenv('GLIMPSE_HIDDEN', 128)),    # hidden size of glimpse fully connected layer

    # REINFORCE
    'STD': float(os.getenv('STD', 0.17)),                       # gaussian policy standard deviation
    'M': int(os.getenv('M', 10)),  # Monte Carlo sampling for valid and test sets

    # model core
    'HIDDEN_SIZE': int(os.getenv('HIDDEN_SIZE', 256)),          # hidden rnn size
    'NUM_GLIMPSES': int(os.getenv('NUM_GLIMPSES', 8))
}

data_transform = transforms.Compose([
    # transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0., 0., 0.],
                         std=[1., 1., 1.])
])

train_data_set = WindTurbinesDataset(
    root_dir=hp['ROOT_DIR'],
    csv_file=hp['ROOT_DIR'] + '/metadata.csv',
    transform=data_transform,
    subset='train',
)

validate_data_set = WindTurbinesDataset(
    root_dir=hp['ROOT_DIR'],
    csv_file=hp['ROOT_DIR'] + '/metadata.csv',
    transform=data_transform,
    subset='validate',
)

hp['NUM_TRAIN'], hp['NUM_VALID'] = len(train_data_set), len(validate_data_set)
train_indices, valid_indices = list(range(hp['NUM_TRAIN'])), list(range(hp['NUM_VALID']))
np.random.shuffle(train_indices)
np.random.shuffle(valid_indices)
train_sampler, valid_sampler = SubsetRandomSampler(train_indices), SubsetRandomSampler(valid_indices)
train_loader = torch.utils.data.DataLoader(
    train_data_set,
    batch_size=hp['BATCH_SIZE'],
    sampler=train_sampler,
    num_workers=1,      # GPU mode
    pin_memory=True,    # GPU mode
)
valid_loader = torch.utils.data.DataLoader(
    validate_data_set,
    batch_size=hp['VALIDATE_BATCH_SIZE'],
    sampler=valid_sampler,
    num_workers=1,      # GPU mode
    pin_memory=True,    # GPU mode
)

model = RecurrentAttention(
    hp['PATCH_SIZE'], hp['NUM_PATCHES'], hp['GLIMPSE_SCALE'],
    hp['NUM_CHANNELS'], hp['LOC_HIDDEN'], hp['GLIMPSE_HIDDEN'],
    hp['STD'], hp['HIDDEN_SIZE'], hp['NUM_CLASSES'],
)
model.cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

tensorboard_dir = os.path.join(hp['LOGS_DIR'], hp['MODEL_NAME'])
print('[*] Saving tensorboard logs to {}'.format(tensorboard_dir))
if not os.path.exists(tensorboard_dir):
    os.makedirs(tensorboard_dir)
configure(tensorboard_dir)

best_valid_acc, patience_counter = 0, 0

for epoch in range(0, hp['EPOCHS']):

    print(
        '\nEpoch: {}/{} - LR: {:.6f}'.format(
            epoch+1, hp['EPOCHS'], hp['LEARNING_RATE'])
    )

    # train for 1 epoch
    train_loss, train_acc = train_one_epoch(model, optimizer, train_loader, epoch, hp)

    # evaluate on validation set
    valid_loss, valid_acc = validate(model, valid_loader, epoch, hp)

    # # reduce lr if validation loss plateaus
    # self.scheduler.step(valid_loss)

    is_best = valid_acc > best_valid_acc
    msg1 = "train loss: {:.3f} - train acc: {:.3f} "
    msg2 = "- val loss: {:.3f} - val acc: {:.3f}"
    if is_best:
        patience_counter = 0
        msg2 += " [*]"
    msg = msg1 + msg2
    print(msg.format(train_loss, train_acc, valid_loss, valid_acc))

    # check for improvement
    if not is_best:
        patience_counter += 1

    if patience_counter > hp['PATIENCE']:
        print("[!] No improvement in a while, stopping training.")
        break
    best_valid_acc = max(valid_acc, best_valid_acc)
