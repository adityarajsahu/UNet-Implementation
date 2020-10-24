#!/usr/bin/env python

import tensorflow as tf
import cv2
import numpy as np

def image_normalize(image, mask):
    # normalize each pixel in the image to bring them within
    # the range [0, 1]
    input_image = tf.constant(image, tf.uint8) / 255
    
    # initialize pixel value lies within the set {1, 2, 3}
    # final pixel value lies within the set {0, 1, 2}
    # scaling down the pixel values results in faster computation
    input_mask = tf.constant(mask, tf.uint8) - 1
    
    return input_image, input_mask

def image_modifier(datapoint):
    # normalize the training images and their corresponding masks
    image, mask = image_normalize(datapoint['image'], datapoint['segmentation_mask'])
    
    # convert the tensor into image for resizing
    #image = tf.image.convert_image_dtype(image, dtype=tf.uint8, saturate=False)
    #mask = tf.image.convert_image_dtype(mask, dtype=tf.uint8, saturate=False)
    
    # resize the images and masks to a particular size to avoid errors
    # in case the size of the images are different
    image = np.resize(image, (128, 128))
    mask = np.resize(mask, (128, 128))
    return image, mask

def processor(dataset):
    
    # Process the training set to get required images and masks.
    train = list(map(image_modifier, dataset['train']))
    test = list(map(image_modifier, dataset['test']))
    
    return train, test

