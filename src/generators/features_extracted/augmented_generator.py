
from preprocess_data.images import (
    load_image,
    preprocess_image_inception,
    get_inception_pretrained,
    rotate_or_flip
)

import os
from tqdm import tqdm
import tensorflow as tf
from pickle import load, dump
from sklearn.utils import shuffle
import numpy as np

from generators.fine_tuned.augmented_generator import FineTunedAugmentedGenerator


class FeaturesExtractedAugmentedGenerator(FineTunedAugmentedGenerator):

    def __init__(self, raw_dataset):
        super().__init__(raw_dataset)
        self.extractor_features_model = get_inception_pretrained()
        # futuro: para nao ser inception podes mandar no init argm model

    def get_image(self, img_name):
        img = self.dict_name_img[img_name]
        img = super().augment_image(img)
        img = preprocess_image_inception(img)

        features = self.extractor_features_model(img)
        img_tensor = tf.reshape(features, [-1])

        return img_tensor
