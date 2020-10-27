#!/usr/bin/env python

import tensorflow as tf
from sklearn.utils import shuffle
import numpy as np
import os


def image_normalize(image, mask):
    input_image = tf.cast(image, tf.float32) / 255.0
    input_mask = tf.cast(mask, tf.float32) - 1.0
    
    return input_image, input_mask


def image_modifier(img_file_path, mask_file_path):
    image = tf.io.read_file(img_file_path)
    image = tf.io.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [128,128])
    
    mask = tf.io.read_file(mask_file_path)
    mask = tf.io.decode_jpeg(mask, channels=1)
    mask = tf.image.resize(mask, [128,128])
    
    image, mask = image_normalize(image, mask)
    
    return image, mask


def processor(img_file_path, mask_file_path):
    image_path, mask_path = shuffle(img_file_path, mask_file_path, random_state=0)
    train_image, train_mask = image_path[:6000], mask_path[:6000]
    val_image, val_mask = image_path[6000:7000], mask_path[6000:7000]
    test_image, test_mask = image_path[7000:], mask_path[7000:]
    
    trainloader = tf.data.Dataset.from_tensor_slices((train_image, train_mask))
    valloader = tf.data.Dataset.from_tensor_slices((val_image, val_mask))
    testloader = tf.data.Dataset.from_tensor_slices((test_image, test_mask))
    
    trainloader = (trainloader.shuffle(32).map(image_modifier, num_parallel_calls=4).batch(8))
    
    valloader = (valloader.shuffle(32).map(image_modifier, num_parallel_calls=4).batch(8))
    
    testloader = (testloader.shuffle(32).map(image_modifier, num_parallel_calls=4).batch(8))
    
    return trainloader, valloader, testloader

