

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

from generators.abstract_generator import Generator, PATH, IMAGE_PATH

from abc import ABC, abstractmethod


class FineTunedGenerator(Generator):

    def __init__(self, raw_dataset):
        self.dict_name_img = self.get_images_loaded(
            raw_dataset,
            file_of_dumped_features="/Users/RitaRamos/Documents/INESC-ID/remote-sensing/notebooks/loaded_image_features.pkl")

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
