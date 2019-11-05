
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

from generators.fine_tuned.abstract_generator import FineTunedGenerator


class FineTunedAugmentedGenerator(FineTunedGenerator):

    def get_image(self, img_name):
        # ver se usas o dict_name_img aqui ou fora
        print("ola")
        img = self.dict_name_img[img_name]
        img = self.augment_image(img)
        return img

    def augment_image(self, img):
        # first_transformation
        # img = change_temperature(img)  # What is patch_size??

        # second_transformation
        img = rotate_or_flip(img)

        # preprocess according to inception
        img = preprocess_image_inception(img)

        return img

# Generator
# FineTuner    SimpleFeatureExtracted
# Simple vs Augmented

        # FineTunerExtractor
