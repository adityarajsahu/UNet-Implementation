import tensorflow as tf
from tf.keras.layers import Conv2D, MaxPool2D, UpSampling2D, Concatenate

class UnetModel:
    def __init__(self):
        super(UnetModel, self).__init__()
    
    # Convolution layer with 64 filters, 3 x 3 kernel, padding = True and Relu               activation function
        self.conv1 = Conv2D(filters=64, kernel_size=3, padding='same', activation='relu')
    # Convolution layer with 128 filters, 3 x 3 kernel, padding = True and Relu activation function
        self.conv2 = Conv2D(filters=128, kernel_size=3, padding='same', activation='relu')
    # Convolution layer with 256 filters, 3 x 3 kernel, padding = True and Relu activation function
        self.conv3 = Conv2D(filters=256, kernel_size=3, padding='same', activation='relu')
    # Convolution layer with 512 filters, 3 x 3 kernel, padding = True and Relu activation function
        self.conv4 = Conv2D(filters=512, kernel_size=3, padding='same', activation='relu')
    # Convolution layer with 1024 filters, 3 x 3 kernel, padding = True and Relu activation function
        self.conv5 = Conv2D(filters=1024, kernel_size=3, padding='same', activation='relu')
    # Convolution layer with 1 filter, 1 x 1 kernel, padding = True and Relu activation function
        self.conv10 = Conv2D(filters=1, kernel_size=1, activation='sigmoid')
    
    # Up Sampling layer with kernel size = 2 x 2
        self.deconv = UpSampling2D(size = (2,2))
    
    # Pooling layer with kernel size = 2 x 2
        self.pool = MaxPool2D(pool_size=(2,2))
    
    def forward(self, inputs, training=False):
    
    # input is the input image to be processed
        conv1_layer = self.conv1(inputs)
        conv1_layer = self.conv1(conv1_layer)
        pool1 = self.pool(conv1_layer)
    
        conv2_layer = self.conv2(pool1)
        conv2_layer = self.conv2(conv2_layer)
        pool2 = self.pool(conv1_layer)
    
        conv3_layer = self.conv3(pool2)
        conv3_layer = self.conv3(conv3_layer)
        pool3 = self.pool(conv3_layer)
    
        conv4_layer = self.conv4(pool3)
        conv4_layer = self.conv4(conv4_layer)
        pool4 = self.pool(conv4_layer)
    
        conv5_layer = self.conv5(pool4)
        conv5_layer = self.conv5(conv5_layer)
    
        up6_layer = self.deconv(conv5_layer)
        merge6 = Concatenate()([up6_layer, conv4_layer])
        conv6_layer = self.conv4(merge6)
        conv6_layer = self.conv4(conv6_layer)
    
        up7_layer = self.deconv(conv6_layer)
        merge7 = Concatenate()([up7_layer, conv3_layer])
        conv7_layer = self.conv3(merge7)
        conv7_layer = self.conv3(conv7_layer)
    
        up8_layer = self.deconv(conv7_layer)
        merge8 = Concatenate()([up8_layer, conv2_layer])
        conv8_layer = self.conv2(merge8)
        conv8_layer = self.conv2(conv8_layer)
    
        up9_layer = self.deconv(conv8_layer)
        merge9 = Concatenate()([up9_layer, conv1_layer])
        conv9_layer = self.conv1(merge9)
        conv9_layer = self.conv1(conv9_layer)
    
        if training:
            conv10_layer = self.conv10(conv9_layer)
    
        return conv10_layer