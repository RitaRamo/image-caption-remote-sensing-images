
from preprocess_data.images import (
    load_image,
    preprocess_image_inception,
    get_inception_pretrained,
    rotate_or_flip,
    preprocess_image,
    get_extractor_features_model
)

import os
from tqdm import tqdm
import tensorflow as tf
from pickle import load, dump
from sklearn.utils import shuffle
import numpy as np

from generators.fine_tuned.augmented_generator import FineTunedAugmentedGenerator
from generators.features_extracted.features import extract_features


class FeaturesExtractedAugmentedGenerator(FineTunedAugmentedGenerator):

    def __init__(self, raw_dataset, image_model_type):
        super().__init__(raw_dataset, image_model_type)
        self.extractor_features_model = get_extractor_features_model(
            self.image_model_type)

    def get_image(self, img_name):
        img = super().get_image(img_name)
        features = extract_features(self.extractor_features_model, img)

        return features


# Before

        # # shape ==  (1, 8, 8, 2048)
        # features = self.extractor_features_model(img)

        # # shape ==  (1, 64, 2048)
        # features = tf.reshape(
        #     features,
        #     (
        #         features.shape[0],
        #         -1,
        #         features.shape[3]
        #     )
        # )

        # # [0] to remove first dim [1, 64, 2048] -> [64, 2048]
        # return features[0]
