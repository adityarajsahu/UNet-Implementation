#!/usr/bin/env python

import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Conv2D, MaxPool2D, UpSampling2D, concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy

def UnetModel(input_size=(128,128,3), output_channel=3):
    input = Input(input_size)
    
    conv_layer1, pool1 = ConvLayer(input, 32, (3,3), (2,2), 'relu', 'same', True)
    conv_layer2, pool2 = ConvLayer(pool1, 64, (3,3), (2,2), 'relu', 'same', True)
    conv_layer3, pool3 = ConvLayer(pool2, 128, (3,3), (2,2), 'relu', 'same', True)
    conv_layer4, pool4 = ConvLayer(pool3, 256, (3,3), (2,2), 'relu', 'same', True)
    
    conv_layer5 = ConvLayer(pool4, 512, (3,3), (2,2), 'relu', 'same', False)
    
    conv_layer6 = UpConvLayer(conv_layer5, 256, (3,3), (2,2), 'relu', 'same', conv_layer4)
    conv_layer7 = UpConvLayer(conv_layer6, 128, (3,3), (2,2), 'relu', 'same', conv_layer3)
    conv_layer8 = UpConvLayer(conv_layer7, 64, (3,3), (2,2), 'relu', 'same', conv_layer2)
    conv_layer9 = UpConvLayer(conv_layer8, 32, (3,3), (2,2), 'relu', 'same', conv_layer1)
    
    output = Conv2D(filters=3, kernel_size=(1,1), activation='softmax', padding='same')(conv_layer9)
    
    model = Model(inputs = input, outputs = output)
    adam = Adam(lr=1e-3)
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    
    return model
    
def ConvLayer(input, filters, kernel_size, pool_size, activation_func, padding, isPool):
    conv = Conv2D(filters=filters, kernel_size=kernel_size, activation=activation_func, padding=padding)(input)
    conv = Conv2D(filters=filters, kernel_size=kernel_size, activation=activation_func, padding=padding)(conv)
    
    if isPool:
        pool = MaxPool2D(pool_size)(conv)
        return conv, pool
    else:
        return conv
    
def UpConvLayer(input, filters, kernel_size, trans_kernel, activation_func, padding, concat_layer):
    layer = UpSampling2D(size=trans_kernel)(input)
    layer = concatenate([layer, concat_layer], axis=3)
    layer = Conv2D(filters=filters, kernel_size=kernel_size, activation=activation_func, padding=padding)(layer)
    layer = Conv2D(filters=filters, kernel_size=kernel_size, activation=activation_func, padding=padding)(layer)
    
    return layer