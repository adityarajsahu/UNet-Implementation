#!/usr/bin/env python

import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import os
import numpy as np
import matplotlib.pyplot as plt
from dataProcessor import processor
from unetmodel import UnetModel

image_path = sorted([os.path.join('images/', filename) for filename in os.listdir('images/') if filename.endswith('.jpg')])

mask_path = sorted([os.path.join('annotations/trimaps/', filename) for filename in os.listdir('annotations/trimaps/') if (filename.endswith('.png') and not filename.startswith('.'))])

trainData, valData, testData = processor(image_path, mask_path)

tf.keras.backend.clear_session()

model = UnetModel()

earlystopping = EarlyStopping(patience=3)

checkpoint_path = 'training_1/cp.ckpt'
checkpoint_dir = os.path.dirname(checkpoint_path)
callback = ModelCheckpoint(filepath=checkpoint_path, save_best_only=True)

model.fit(trainData, epochs=10, validation_data=valData, callbacks=[earlystopping, callback])

test_image, test_mask = next(iter(testData))

prediction = model.predict(test_image)
prediction = np.argmax(prediction, axis=-1)
prediction = np.expand_dims(prediction, axis=-1)

figure, axis = plt.subplots(nrows=3, ncols=3, figsize=(9,9))
  
for i in range(3):
  axs[i][0].imshow(test_image[i]);
  axs[i][1].imshow(np.squeeze(test_mask[i],-1), cmap='gray')
  axs[i][2].imshow(np.squeeze(prediction[i],-1), cmap='gray')