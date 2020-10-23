#!/usr/bin/env python

from unet-model import MyModel
from dataProcessor import processor
from tf.keras.callbacks import ModelCheckpoint

# Downloading Oxford - IIIT Pet dataset for image segmentation 
dataset, info = tfds.load('oxford_iiit_pet:3.*.*', with_info=True)

"""
The Oxford-IIIT pet dataset is a 37 category pet image dataset with 
roughly 200 images for each class.The images have large variations 
in scale, pose and lighting. All images have an associated ground 
truth annotation of breed.

In this dataset, the output is also an image in which ever pixel is 
assigned to a particular class -
1. Pixel 1 - pixel inside the periphery of a pet body.
2. Pixel 2 - pixel on the periphery of the pet body.
3. Pixel 3 - pixel outisde the periphery of the pet body.
"""

# Process the downloaded data according to requirement
train, test = processor(dataset)

# create the model  
unet = MyModel()

# initialize the optimizer, losses and metrices
unet.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss=keras.losses.Binaryentropy(), metrics=[keras.metrics.SparseCategoricalAccuracy()])
# initialize the checkpoint for saving model
model_checkpoint_callback = ModelCheckpoint('unet_weight.hdf5', monitor='loss',verbose=1, save_best_only=True)
# train and validate the model
unet.fit(train["image"], train["segmentation_mask"], batch_size=4, epoch=25, validation_data=(test["image"], test["segmentation_mask"]), callbacks=[model_checkpoint_callback])
