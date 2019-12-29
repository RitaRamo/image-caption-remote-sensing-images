
from preprocess_data.images import (
    load_image,
    preprocess_image_inception,
    get_inception_pretrained,
    rotate_or_flip,
    preprocess_image,
    get_extractor_features_model
)

import os
from tqdm import tqdm
import tensorflow as tf
from pickle import load, dump
from sklearn.utils import shuffle
import numpy as np

# from generators.abstract_generator import Generator, PATH  # , IMAGE_PATH
from generators.fine_tuned.simple_generator import FineTunedSimpleGenerator
from generators.features_extracted.features import extract_features


class FeaturesExtractedSimpleGenerator(FineTunedSimpleGenerator):

    def __init__(self, raw_dataset, image_model_type):
        super().__init__(raw_dataset, image_model_type)
        self.extractor_features_model = get_extractor_features_model(
            self.image_model_type)

    def get_image(self, img_name):
        img = super().get_image(img_name)
        features = extract_features(self.extractor_features_model, img)

        return features

# Before

        # # (1, 8, 8, 2048)  ; alternatively: model.predict()
        # features = self.extractor_features_model(img)

        # # (1, 64, 2048) -> -1 is to flatten
        # features = tf.reshape(
        #     features,
        #     (
        #         features.shape[0],
        #         -1,
        #         features.shape[3]
        #     )
        # )

        # # [0] to remove first dim [1, 64, 2048] -> [64,2048]
        # return features[0]

# class FeaturesExtractedSimpleGenerator(FineTunedSimpleGenerator):

#     def __init__(self, raw_dataset, image_model_type):
#         super().__init__(raw_dataset, image_model_type)
#         # self.dict_name_img = self.get_images_loaded(
#         #     raw_dataset,
#         #     file_of_dumped_features="src/generators/loaded_image_features.pkl")
#         self.extractor_features_model = get_extractor_features_model(
#             self.image_model_type)

#     def get_images_loaded(self, raw_dataset, file_of_dumped_features="", path=""):
#         if os.path.exists(file_of_dumped_features):
#             return load(open(file_of_dumped_features, 'rb'))
#         else:
#             dict_name_img = {}
#             images_names = [row["filename"] for row in raw_dataset["images"]]
#             images = set(images_names)
#             for img_filename in tqdm(images):
#                 img_path = self.IMAGE_PATH + img_filename
#                 img_loaded = load_image(img_path)

#                 dict_name_img[img_filename] = img_loaded

#             return dict_name_img

#     def get_image(self, img_name):
#         # # # ver se usas o dict_name_img aqui ou fora
#         # img = self.dict_name_img[img_name]
#         # # # [0] to remove first dim [1, 299, 299, 3]

#         img = load_image(PATH+IMAGE_PATH + img_name)

#         print("\nthis is my img shape loaded", np.shape(img))

#         img = preprocess_image(img, self.image_model_type)
#         print("this is my img shape after pre", np.shape(img))

#         # (1, 8, 8, 2048)  ; alternatively: model.predict()
#         features = self.extractor_features_model(img)
#         print("this is my featyres shape", np.shape(features))

#         # (1, 64, 2048) -> -1 is to flatten
#         features = tf.reshape(
#             features,
#             (
#                 features.shape[0],
#                 -1,
#                 features.shape[3]
#             )
#         )

#         # [0] to remove first dim [1, 64, 2048] -> [64,2048]
#         return features[0]

# TODO: Isto!
# experimentar sem o image loaded!! ->done
# Por time, ver qual é o + rapido eheh!
# sem loaded no dict- MELHOR
# Depois fazer para o outro modelo!
#
# Yeah! Fazer a parte do attention do val e do early_stop!


# class FeaturesExtractedSimpleGenerator(Generator):

#     def __init__(self, raw_dataset, image_model_type):
#         super().__init__(raw_dataset, image_model_type)
#         self.images_features = self.extract_features(
#             raw_dataset,
#             file_of_dumped_features="src/generators/all_image_features.pkl")

#     def extract_features(self, raw_dataset, file_of_dumped_features="", path=PATH+IMAGE_PATH):

#         images_features = {}
#         extractor_features_model = get_extractor_features_model(
#             self.image_model_type)  # get_inception_pretrained()
#         images_names = [row["filename"] for row in raw_dataset["images"]]
#         images = set(images_names)
#         for img_filename in tqdm(images):
#             img_path = path + img_filename
#             img_loaded = load_image(img_path)

#             img = preprocess_image(img_loaded, self.image_model_type)

#             # (1, 8, 8, 2048)  ; alternatively: model.predict()
#             features = extractor_features_model(img)
#             # (1, 64, 2048) -> -1 is to flatten
#             features = tf.reshape(
#                 features,
#                 (
#                     features.shape[0],
#                     -1,
#                     features.shape[3]
#                 )
#             )
#             images_features[img_filename] = features
#         return images_features

#     def get_image(self, img_name):
#         features = self.images_features[img_name]
#         # [0] to remove first dim [1, 64, 2048] -> [64,2048]
#         return features[0]

#         # img_tensor = tf.reshape(features, [-1])
#         # return img_tensor


# class FeaturesExtractedSimpleGenerator(Generator):

#     def __init__(self, raw_dataset, image_model_type):
#         super().__init__(raw_dataset, image_model_type)
#         self.images_features = self.extract_features(
#             raw_dataset,
#             file_of_dumped_features="src/generators/all_image_features.pkl")

#     def extract_features(self, raw_dataset, file_of_dumped_features="", path=PATH+IMAGE_PATH):
#         if os.path.exists(file_of_dumped_features):
#             print("ola entrei aqui")
#             return load(open(file_of_dumped_features, 'rb'))
#         else:
#             print("opa naoo")

#             images_features = {}
#             extractor_features_model = get_inception_pretrained()
#             images_names = [row["filename"] for row in raw_dataset["images"]]
#             images = set(images_names)
#             for img_filename in tqdm(images):
#                 img_path = path + img_filename
#                 img_loaded = load_image(img_path)

#                 img = preprocess_image(img, self.image_model_type)

#                 # (1, 8, 8, 2048)  ; alternatively: model.predict()
#                 features = extractor_features_model(img)
#                 # (1, 64, 2048) -> -1 is to flatten
#                 features = tf.reshape(
#                     features,
#                     (
#                         features.shape[0],
#                         -1,
#                         features.shape[3]
#                     )
#                 )
#         #TODO: ERRADOOOO
#                 images_features[img_filename] = img_loaded
#             return images_features

#     def get_image(self, img_name):
#         features = self.images_features[img_name]
#         # [0] to remove first dim [1, 64, 2048] -> [64,2048]
#         return features[0]

#         # img_tensor = tf.reshape(features, [-1])
#         # return img_tensor
