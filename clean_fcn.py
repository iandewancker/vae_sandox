import tensorflow as tf
import numpy as np
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from PIL import Image
import glob, os
from shutil import copy2
import uuid
from matplotlib import pyplot as plt
import gc
import datetime
import sklearn
from sklearn.externals import joblib
import boto3

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Dense, Dropout, Flatten, Input
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D
from tensorflow.python.keras.optimizers import SGD
from tensorflow.python.keras.layers import *
from tensorflow.python.keras.regularizers import l2
from tensorflow.python.keras.models import model_from_json
import tensorflow


# define keras model
def FCN(input_shape = None, batch_shape=None):

    if batch_shape:
        img_input = Input(batch_shape=batch_shape)
        image_size = batch_shape[1:3]
    else:
        img_input = Input(shape=input_shape)
        image_size = input_shape[0:2]

    bn_axis = 3
    # just one class, unpickable
    classes = 1

    x = Conv2D(256, (10, 10), strides=(1, 1), padding='same', name='conv1')(img_input)
    x = BatchNormalization(axis=bn_axis, name='bn_conv1', momentum=0.9)(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = BatchNormalization(axis=bn_axis, name='bn_conv2', momentum=0.9)(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    #x = BatchNormalization(axis=bn_axis, name='bn_conv1', momentum=batch_momentum)(x)

    #x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    # x = Conv2D(128, (5, 5), strides=(1, 1), padding='same', name='conv3')(x)
    # x = BatchNormalization(axis=bn_axis, name='bn_conv1', momentum=batch_momentum)(x)
    # x = Activation('relu')(x)
    #x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    #x = Conv2D(128, (3, 3), strides=(2, 2), padding='same', name='conv4')(x)
    #x = BatchNormalization(axis=bn_axis, name='bn_conv1', momentum=batch_momentum)(x)
    #x = Activation('relu')(x)
    #x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    #x = Conv2D(128, (5, 5), strides=(2, 2), padding='same', name='conv3')(img_input)
    #x = BatchNormalization(axis=bn_axis, name='bn_conv1', momentum=batch_momentum)(x)
    #x = Activation('relu')(x)
    #x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    #classifying layer
    #x = Conv2D(classes, (3, 3), dilation_rate=(2, 2), kernel_initializer='normal', activation='linear', padding='same', strides=(1, 1), kernel_regularizer=l2(weight_decay))(x)
    x = Conv2D(classes, (1, 1), kernel_initializer='he_normal', activation='sigmoid', padding='same', strides=(1, 1))(x)
    # = BilinearUpSampling2D(target_size=tuple(image_size))(x)
    #x = tf.image.resize_images(x, [image_size[0], image_size[1]])
    x = UpSampling2D(size=(4, 4), data_format=None)(x)

    model = Model(img_input, x)
    # weights_path = os.path.expanduser(os.path.join('~', '.keras/models/fcn_resnet50_weights_tf_dim_ordering_tf_kernels.h5'))
    # model.load_weights(weights_path, by_name=True)
    return model

model = FCN(input_shape=(500,700,2))

# define loss function
def binary_crossentropy_with_logits(ground_truth, predictions):
    return tensorflow.keras.backend.mean(tensorflow.keras.backend.binary_crossentropy(ground_truth,
                                        predictions,
                                        from_logits=True),axis=-1)

def create_weighted_binary_crossentropy(zero_weight, one_weight):
    def weighted_binary_crossentropy(y_true, y_pred):
        # Calculate the binary crossentropy
        b_ce = tensorflow.keras.backend.binary_crossentropy(y_true, y_pred)

        # Apply the weights
        weight_vector = y_true * one_weight + (1. - y_true) * zero_weight
        weighted_b_ce = weight_vector * b_ce

        # Return the mean error
        return tensorflow.keras.backend.mean(weighted_b_ce)
    return weighted_binary_crossentropy


# Train model
model.compile(loss=create_weighted_binary_crossentropy(1.0,1.0),
              optimizer='rmsprop',
              metrics=['binary_accuracy'])

# X_input is (500,700,2)  grayscale and depth
# Y_masks are binary images (500,700,1) where channel is unpickable probability
model.fit(X_input, Y_masks, batch_size=5, epochs=700)