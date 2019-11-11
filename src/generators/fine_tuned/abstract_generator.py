

import os
from abc import ABC, abstractmethod
from pickle import dump, load

import tensorflow as tf
from sklearn.utils import shuffle
from tqdm import tqdm

from generators.abstract_generator import IMAGE_PATH, PATH, Generator
from preprocess_data.images import load_image


class FineTunedGenerator(Generator):

    def __init__(self, raw_dataset):
        super().__init__(raw_dataset)
        self.dict_name_img = self.get_images_loaded(
            raw_dataset,
            file_of_dumped_features="src/generators/loaded_image_features.pkl")

    def get_images_loaded(self, raw_dataset, file_of_dumped_features="", path=PATH+IMAGE_PATH):
        if os.path.exists(file_of_dumped_features):
            return load(open(file_of_dumped_features, 'rb'))
        else:
            dict_name_img = {}
            images_names = [row["filename"] for row in raw_dataset["images"]]
            images = set(images_names)
            for img_filename in tqdm(images):
                img_path = path + img_filename
                img_loaded = load_image(img_path)

                dict_name_img[img_filename] = img_loaded

            return dict_name_img

    @abstractmethod
    def get_image(self, img):
        pass
