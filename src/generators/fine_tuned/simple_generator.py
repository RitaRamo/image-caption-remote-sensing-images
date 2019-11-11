
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


class FineTunedSimpleGenerator(FineTunedGenerator):

    def get_image(self, img_name):
        # ver se usas o dict_name_img aqui ou fora
        img = self.dict_name_img[img_name]
        # [0] to remove first dim [1, 299, 299, 3]
        img = preprocess_image_inception(img)[0]

        return img
