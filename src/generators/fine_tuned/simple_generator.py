
from preprocess_data.images import (
    load_image,
    preprocess_image_inception,
    get_inception_pretrained,
    rotate_or_flip,
    preprocess_image
)

import os
from tqdm import tqdm
import tensorflow as tf
from pickle import load, dump
from sklearn.utils import shuffle

#from generators.fine_tuned.abstract_generator import FineTunedGenerator
from generators.abstract_generator import Generator


class FineTunedSimpleGenerator(Generator):

    def get_image(self, img_name):
        img = load_image(self.IMAGE_PATH + img_name)

        # [0] to remove first dim [1, 299, 299, 3]
        img = preprocess_image(img, self.image_model_type)  # [0]

        return img

# Before
    # def get_image(self, img_name):
    #     img = self.dict_name_img[img_name]
    #     # [0] to remove first dim [1, 299, 299, 3]
    #     img = preprocess_image(img, self.image_model_type)[0]

    #     return img
