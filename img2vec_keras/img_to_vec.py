#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from tensorflow import keras

from tensorflow.keras.applications import mobilenet_v2
from tensorflow.keras.applications import resnet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
import numpy as np


_IMAGE_NET_TARGET_SIZE = (224, 224)


class Img2Vec(object):

    def __init__(self, model_name='resnet'):
        if (model_name == 'mobilenet'):
            print("Using MobileNetV2 to vectorize images")
            model = mobilenet_v2.MobileNetV2(weights='imagenet')
            layer_name = 'global_average_pooling2d'
            self.preprocess_fn = mobilenet_v2.preprocess_input
        else:
            print("Using ResNet50 to vectorize images")
        model = resnet50.ResNet50(weights='imagenet')
        layer_name = 'avg_pool'
            self.preprocess_fn = resnet50.preprocess_input
        o = model.get_layer(layer_name).output
        self.vec_len = o.shape[1]
        self.intermediate_layer_model = Model(inputs=model.input, 
                                              outputs=model.get_layer(layer_name).output)


    def get_vec(self, image_path):
        """ Gets a vector embedding from an image.
        :param image_path: path to image on filesystem
        :returns: numpy ndarray
        """

        img = image.load_img(image_path, target_size=_IMAGE_NET_TARGET_SIZE)
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = resnet50.preprocess_input(x)
        intermediate_output = self.intermediate_layer_model.predict(x)
        
        return intermediate_output[0]
    
if __name__ == "main":
     pass    