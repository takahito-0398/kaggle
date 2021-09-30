# import keras
import os
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix

# from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from keras.models import Sequential, Model
from keras.layers import Flatten, Dense,Input, Dropout
import keras
from keras_applications.resnext import ResNeXt101

from keras import optimizers
from tensorflow.python.client import device_lib
import tensorflow as tf
from keras_applications.resnext import ResNeXt101

import efficientnet.keras
import keras_efficientnet_v2

from _00_utils import roc_auc

def myResNet50V2(input_shape):
    from keras.applications import ResNet50V2
    input_tensor = Input(shape=input_shape)
    # ResNet50 = ResNet50(include_top=False, weights='imagenet',input_tensor=input_tensor)
    main_model = ResNet50V2(include_top=False, weights='imagenet',input_tensor=input_tensor)

    top_model = Sequential()
    top_model.add(Flatten(input_shape=main_model.output_shape[1:]))
    top_model.add(Dense(256, activation='relu'))
    top_model.add(Dropout(0.5))
    # 2値分類なので2
    top_model.add(Dense(2, activation='softmax'))

    top_model = Model(input=main_model.input, output=top_model(main_model.output))
    top_model.compile(loss='categorical_crossentropy',optimizer=optimizers.SGD(lr=1e-3, momentum=0.9),metrics=['accuracy', roc_auc])

    return top_model

def myResNeXt101(input_shape):

    input_tensor = Input(shape=input_shape)
    main_model = ResNeXt101(weights='imagenet',
                    backend=keras.backend,
                    layers=keras.layers,
                    models=keras.models,
                    utils=keras.utils,
                    include_top=False,
                    input_tensor=input_tensor)
    # ResNet50 = ResNet50(include_top=False, weights='imagenet',input_tensor=input_tensor)
    # main_model = ResNeXt101(include_top=False, weights='imagenet',input_tensor=input_tensor)

    top_model = Sequential()
    top_model.add(Flatten(input_shape=main_model.output_shape[1:]))
    top_model.add(Dense(256, activation='relu'))
    top_model.add(Dropout(0.5))
    # 2値分類なので2
    top_model.add(Dense(2, activation='softmax'))

    top_model = Model(input=main_model.input, output=top_model(main_model.output))
    top_model.compile(loss='categorical_crossentropy',optimizer=optimizers.SGD(lr=1e-3, momentum=0.9),metrics=['accuracy', roc_auc])

    return top_model

def myEfficientNetV2B0(input_shape):
    import efficientnet.keras


    # main_model = keras_efficientnet_v2.EfficientNetV2S(pretrained="imagenet")
    # input_tensor = Input(shape=input_shape) 
    # main_model = keras_efficientnet_v2.EfficientNetV2B0(dropout=1e-6, num_classes=0, pretrained="imagenet21k", input_tensor=input_tensor)
    main_model = keras_efficientnet_v2.EfficientNetV2M(input_shape=input_shape, num_classes=0, pretrained="imagenet21k-ft1k")



    top_model = Sequential()
    top_model.add(Flatten(input_shape=main_model.output_shape[1:]))
    top_model.add(Dense(256, activation='relu'))
    top_model.add(Dropout(0.5))
    # 2値分類なので2
    top_model.add(Dense(2, activation='softmax'))

    top_model = Model(input=main_model.input, output=top_model(main_model.output))
    top_model.compile(loss='categorical_crossentropy',optimizer=optimizers.SGD(lr=1e-3, momentum=0.9),metrics=['accuracy', roc_auc])

    return top_model


# model = tf.keras.models.Sequential([
#     tf.keras.layers.InputLayer(input_shape=[224, 224, 3]),
#     effnetv2_model.get_model('efficientnetv2-b0', include_top=False, pretrained=True),
#     tf.keras.layers.Dropout(rate=0.2),
#     tf.keras.layers.Dense(4, activation='softmax'),
# ])

# model = keras_efficientnet_v2.EfficientNetV2M(input_shape=input_shape, num_classes=0, pretrained="imagenet21k-ft1k")

# print(model(np.ones([1, 224, 224, 3])).shape)
# # (1, 7, 7, 1280)
# print(model(np.ones([1, 512, 512, 3])).shape)
# # (1, 16, 16, 1280)
