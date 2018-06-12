import os
from datetime import datetime
from time import time

from keras import backend as K, Sequential
from keras.callbacks import TensorBoard, EarlyStopping
from keras.layers import Conv2D, Activation, MaxPooling2D, Flatten, Dropout, Dense
from keras.optimizers import RMSprop, Adam
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from sklearn.metrics import accuracy_score

SCRIPT_VERSION = '0.0.7'
SCRIPT_NAME = os.path.basename(__file__)
SCRIPT_START = time()
TIMESTAMP = str(datetime.now()).replace(':', '.')
SIGNATURE = SCRIPT_NAME + ' ' + TIMESTAMP
DATA_DIR = os.getenv('DATA_DIR', '../data/windturbines/')

K.set_floatx('float16')  # reduce memory size

# Hyperparameters
hp = {
    'IMG_WIDTH': 512,
    'IMG_HEIGHT': 512,
    'TRAIN_TEST_SPLIT': 0.1,
    'BATCH_SIZE': int(os.getenv('BATCH_SIZE', 20)),
    'TRAIN_VALIDATE_SPLIT': float(os.getenv('TRAIN_VALIDATE_SPLIT', 0.1)),
    'EPOCHS': int(os.getenv('EPOCHS', 200)),
    'LEARNING_RATE': float(os.getenv('LEARNING_RATE', 1e-3)),
    'DROPOUT': float(os.getenv('DROPOUT', 0.25)),
    'EARLY_STOPPING': bool(os.getenv('EARLY_STOPPING', False)),
    'PATIENCE': int(os.getenv('PATIENCE', 16)),
}
OPTIMIZER = Adam(lr=hp['LEARNING_RATE'])

# with open(DATA_DIR + 'metadata.csv') as f:
#     reader = csv.DictReader(f)
#     for index, row in enumerate(reader):
#         if not path.isfile(row['image_file']):
#             print('Warning: don\'t have {} as listed in {} on row {}'.format(
#                 row['image_file'],
#                 DATA_DIR + 'metadata.csv',
#                 index + 2))

# Normalization initialisation
train_datagen = ImageDataGenerator()
normalization_generator = train_datagen.flow_from_directory(
    DATA_DIR + 'train',
    target_size=(hp['IMG_WIDTH'], hp['IMG_HEIGHT']),
    batch_size=200
)
norm_sample_X, _ = next(normalization_generator)  # normalization sample from the set
# re-declare to use normalization sample
train_datagen = ImageDataGenerator(
    horizontal_flip=True,
    vertical_flip=True,
    samplewise_std_normalization=True,
    samplewise_center=True,
    # zca_whitening=True,
    validation_split=0.1,
)
train_datagen.fit(norm_sample_X, augment=True)

train_generator = train_datagen.flow_from_directory(
    DATA_DIR + 'train',
    subset='training',
    target_size=(hp['IMG_WIDTH'], hp['IMG_HEIGHT']),
    batch_size=hp['BATCH_SIZE']
)

validation_generator = train_datagen.flow_from_directory(
    DATA_DIR + 'validate',
    subset='validation',
    target_size=(hp['IMG_WIDTH'], hp['IMG_HEIGHT']),
    batch_size=hp['BATCH_SIZE'],
    class_mode='categorical')

if K.image_data_format() == 'channels_first':
    input_shape = (3, hp['IMG_WIDTH'], hp['IMG_HEIGHT'])
else:
    input_shape = (hp['IMG_WIDTH'], hp['IMG_HEIGHT'], 3)

model = Sequential()
model.add(Conv2D(32, (4, 4), input_shape=input_shape))
model.add(Activation('lrelu'))
model.add(MaxPooling2D(pool_size=(3, 3)))

model.add(Conv2D(48, (4, 4)))
model.add(Activation('lrelu'))
model.add(MaxPooling2D(pool_size=(3, 3)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('lrelu'))
model.add(MaxPooling2D(pool_size=(3, 3)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('lrelu'))
model.add(MaxPooling2D(pool_size=(3, 3)))

model.add(Conv2D(96, (3, 3)))
model.add(Activation('lrelu'))
model.add(MaxPooling2D(pool_size=(3, 3)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(2))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer=OPTIMIZER,
              metrics=['accuracy'])
model.summary()

callbacks = [TensorBoard(log_dir='./tensorboard_log/' + SIGNATURE, write_graph=False)]
if hp['EARLY_STOPPING']:
    callbacks.append(EarlyStopping(patience=hp['PATIENCE'], min_delta=1e-3))


history = model.fit_generator(
    train_generator,
    epochs=hp['EPOCHS'],
    steps_per_epoch=96,
    validation_data=validation_generator,
    validation_steps=16,
    callbacks=callbacks,
    # workers=7,
).history

# Run on unseen test data
test_imgs, test_labels = validation_generator
test_pred = [np.argmax(prediction) for prediction in model.predict(test_imgs)]
test_labels = [np.argmax(label) for label in test_labels]
accuracy = accuracy_score(test_labels, test_pred)
print(accuracy)
