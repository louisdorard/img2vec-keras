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
                                              outputs=o)

    def path2img(self, image_path):
        return image.load_img(image_path, target_size=_IMAGE_NET_TARGET_SIZE)

    def img_to_array(self, img):
        return image.img_to_array(img)

    def img_preprocess(self, img):
        img_processed = img.copy()
        # if we're given just one image, we need to expand dims; otherwise we go straight to preprocessing
        if len(img_processed.shape) < 4:
            img_processed = np.expand_dims(img_processed, axis=0)
        img_processed = self.preprocess_fn(img_processed)
        return img_processed

    def img_processed_to_vec(self, img_processed):
        intermediate_output = self.intermediate_layer_model.predict(img_processed)
        return intermediate_output

    def img2vec(self, img):
        return self.img_processed_to_vec(self.img_preprocess(self.img_to_array(img)))

    def path2vec(self, image_path):
        """ Gets a vector embedding from an image, given the path to the image file.
        :param image_path: path to image on filesystem
        :returns: numpy ndarray
        """
        return self.img2vec(self.path2img(image_path))

if __name__ == "main":
     pass    