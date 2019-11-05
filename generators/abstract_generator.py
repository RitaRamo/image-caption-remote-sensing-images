
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

from abc import ABC, abstractmethod

PATH = "datasets/RSICD/"
IMAGE_PATH = "/RSICD_images/"


class Generator(ABC):

    def __init__(self, raw_dataset):
        pass

    @abstractmethod
    def get_image(self, img):
        pass

    def generate(self, train_images_names, input_captions, target_captions, vocab_size):

        dataset_size = len(train_images_names)

        while True:
            for i in range(dataset_size):

                img_name = train_images_names[i]
                img_tensor = self.get_image(img_name)

                caption_input_tensor = input_captions[i]
                caption_target_tensor = tf.keras.utils.to_categorical(
                    target_captions[i],
                    num_classes=vocab_size
                )

                yield {
                    'input_1': img_tensor,
                    'input_2': caption_input_tensor
                }, caption_target_tensor


# AugmentedFineTuneGenerator
# AugmentedFeatureExtractorGenerator
# SimpleFineTuneGenerator
# SimpleFeatureExtractorGenerator

# FineTune
    # augmented_generator -> # FineTunedAugmentedGenerator
    # simple_generator -># FineTunedSimpleGenerator

#FeatureExtractor (test)
    # #augmented_generator -> FeaturesExtractedAugmentedGenerator
    # #simple_generator -> FeaturesExtractedSimpleGenerator


# Evaluator
    # def convert_img_for_test_using_feature_extraction()
    # def convert_img_for_test_using_finetune()


# agumented
# simple->simple1 (convert_img_for_test_using_feature_extraction); simple2 (convert_img_for_test_using_finetune)
# augmented-> 1 (convert_img_for_test_using_feature_extraction) and 2 (convert_img_for_test_using_finetune)