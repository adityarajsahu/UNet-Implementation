#!/usr/bin/env python

import tensorflow as tf
import tensorflow_datasets as tfds
import cv2

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

def image_normalize(image, mask):
    # normalize each pixel in the image to bring them within
    # the range [0, 1]
    input_image = tf.constant(image, tf.float16) / 255.0
    
    # initialize pixel value lies within the set {1, 2, 3}
    # final pixel value lies within the set {0, 1, 2}
    # scaling down the pixel values results in faster computation
    input_mask = tf.constant(mask, tf.float16) - 1
    
    return input_image, input_mask

def image_modifier(datapoint):
    # normalize the training images and their corresponding masks
    image, mask = image_normalize(datapoint['image'], datapoint['segmentation_mask'])
    
    # convert the tensor into image for resizing
    image = tf.image.convert_image_dtype(image, dtype=tf.float16, saturate=False)
    mask = tf.image.convert_image_dtype(mask, dtype=tf.float16, saturate=False)
    
    # resize the images and masks to a particular size to avoid errors
    # in case the size of the images are different
    image = cv2.resize(image, (128, 128))
    mask = cv2.resize(mask, (128, 128))
    return image, mask

def processor(dataset):
    
    # Process the training set to get required images and masks.
    train = map(image_modifier, dataset['train'])
    test = map(image_modifier, dataset['test'])
    
    return train, test

