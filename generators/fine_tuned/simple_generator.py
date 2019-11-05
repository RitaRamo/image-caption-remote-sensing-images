
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

    def get_image(self, img_name, dict_name_img):
        # ver se usas o dict_name_img aqui ou fora
        img = self.dict_name_img[img_name]
        img = preprocess_image_inception(img)

        return img


# FineTunedSimple:
    # load images + load preprocess image                (se for ler logo__ test difere)
    # test -> igual

# FineTunedAugmented:
    # load images + augment image + preprocess image      +(se for feature_extraction+__ test difere)
    # test (#load images + load preprocess image  )


# FeatureExtracot
    # images_features()
    # load images + agument + preprocess image + feature extraction

    # test (load images + load process image + feature extraction)


# AugmentedFineTuneGenerator
# AugmentedFeatureExtractorGenerator
# SimpleFineTuneGenerator
# SimpleFeatureExtractorGenerator

# FineTuneEvaluator
# def convert_img_for_test():

# FeturesExtractedEvaluator
# def convert_img_for_test()
