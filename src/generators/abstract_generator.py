
from preprocess_data.images import (
    load_image,
    rotate_or_flip
)

import os
from tqdm import tqdm
import tensorflow as tf
from pickle import load, dump
from sklearn.utils import shuffle
import numpy as np

from abc import ABC, abstractmethod

PATH = "src/datasets/RSICD/raw_dataset/"
#IMAGE_PATH = "/RSICD_images/"


class Generator(ABC):
    IMAGE_PATH = PATH + "RSICD_images/"

    def __init__(self, raw_dataset, image_model_type):
        self.raw_dataset = raw_dataset
        self.image_model_type = image_model_type

    @abstractmethod
    def get_image(self, img_name):
        pass

    def get_shape_of_input_image(self):
        any_img_name = self.raw_dataset["images"][0]["filename"]
        return np.shape(self.get_image(any_img_name)[0])

    def generate(self, train_images_names, input_captions, target_captions, vocab_size):

        dataset_size = len(train_images_names)

        while True:
            for i in range(dataset_size):

                img_name = train_images_names[i]
                img_tensor = self.get_image(img_name)

                # [0] to remove first dim, ex for inception:
                # fine_tuned image shape: [1, 299, 299, 3] -> [299, 299, 3]
                # feature_extracted image shape: [1, 64, 2048] -> [64,2048]
                img_tensor = img_tensor[0]

                caption_input_tensor = input_captions[i]
                caption_target_tensor = tf.keras.utils.to_categorical(
                    target_captions[i],
                    num_classes=vocab_size
                )

                yield {
                    'input_1': img_tensor,
                    'input_2': caption_input_tensor
                }, caption_target_tensor
