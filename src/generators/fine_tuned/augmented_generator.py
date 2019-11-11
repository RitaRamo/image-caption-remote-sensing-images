
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
        img = self.dict_name_img[img_name]
        img = self.augment_image(img)
        img = preprocess_image_inception(img)[0]
        return img

    def augment_image(self, img):
        img = rotate_or_flip(img)
        return img
