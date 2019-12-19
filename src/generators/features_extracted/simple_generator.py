
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

from generators.abstract_generator import Generator, PATH, IMAGE_PATH


class FeaturesExtractedSimpleGenerator(Generator):

    def __init__(self, raw_dataset):
        super().__init__(raw_dataset)
        self.images_features = self.extract_features(
            raw_dataset,
            file_of_dumped_features="src/generators/all_image_features.pkl")

    def extract_features(self, raw_dataset, file_of_dumped_features="", path=PATH+IMAGE_PATH):
        if os.path.exists(file_of_dumped_features):
            return load(open(file_of_dumped_features, 'rb'))
        else:
            images_features = {}
            extractor_features_model = get_inception_pretrained()
            images_names = [row["filename"] for row in raw_dataset["images"]]
            images = set(images_names)
            for img_filename in tqdm(images):
                img_path = path + img_filename
                img_loaded = load_image(img_path)
                img = preprocess_image_inception(img_loaded)

                # (1, 8, 8, 2048)  ; alternatively: model.predict()
                features = extractor_features_model(img)
                # (1, 64, 2048) -> -1 is to flatten
                features = tf.reshape(
                    features,
                    (
                        features.shape[0],
                        -1,
                        features.shape[3]
                    )
                )

                images_features[img_filename] = img_loaded
            return images_features

    def get_image(self, img_name):
        features = self.images_features[img_name]
        # [0] to remove first dim [1, 64, 2048] -> [64, 2048]
        # return features[0]

        img_tensor = tf.reshape(features, [-1])
        return img_tensor
