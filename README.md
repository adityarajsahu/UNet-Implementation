# UNet-Implementation

UNet is a image classification model based on semantic segmentation. The goal of semantic image segmentation is to label each pixel of an image with a corresponding class of what is being represented. Because we�re predicting for every pixel in the image, this task is commonly referred to as dense prediction. 

## UNet Architecture

The UNet model is based on encoder - decoder operation, where the input image is initially downsampled using Convolutional Neural Networks. Then, the downsampled image is again upsampled inorder to match it's feature vectors to that of the input mask. Here is the [link](https://arxiv.org/pdf/1505.04597.pdf) to the research paper published on UNet.

![img/architecture.png](img/architecture.png)

Here is the image representing the actual architecture of UNet, as published in the research paper.
Input - Image _[size : 572 x 572 pixels]_
Output - Image (Mask) _[size : 388 x 388 pixels]_

## Dataset 

The dataset taken for this model training and validation is **Oxford - IIIT Pet** dataset. For downloading the dataset, type the commands that are mentioned below in your terminal :-
```
curl -O http://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz
curl -O http://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz
tar -xf images.tar.gz
tar -xf annotations.tar.gz
```

The folder _images/_ contains the input images and _annotations/trimaps/_ conatains output masks.
For this project, I have resized the images and masks to _128 x 128 pixels_ for this model.

![img/image_mask.png](img/image_mask.png)

Every pixel in the mask is classified into one of the three categories :-
- pixel outside the periphery of the body of animal.
- pixel on the periphery of the body of animal.
- pixel inside the periphery of the body of animal.

## Dependencies 

Some packages are required for running the codes in the repository. To install them write the following commands in the terminal :-

```
pip install -r requirements.text
```

Apart from this, we also need to install h5py package to save the weights after training.

```
pip install -q pyyaml h5py
```

## Content of the Repository

- **__pycache__** : _Contains the bytecode generated by the interpreter._
- **annotations** : _Contains the masks for training, validating and testing our model._
- **images** : _Conatins the images for training, validating and testing our model._
- **img** : _Conatins images for README.md file._
- **training_1/cp.ckpt** : _Conatins the saved weights after training the model._
- **LICENSE** : _License documentation._
- **README.md** : _Documentation of this repository._
- **dataProcessor.py** - _Code for preprocessing our input data before training our model._
- **modelTrainer.py** - _Code to train the model and save the weights._
- **requirements.txt** - _Used to install dependencies._
- **unetmodel.py** - _Code of UNet model written from scratch._
