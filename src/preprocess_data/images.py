import tensorflow as tf
from enum import Enum
import numpy as np
from scipy.interpolate import UnivariateSpline
import logging


def load_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    return img


def preprocess_image_inception(img):  # according to inception
    img = tf.image.resize(img, (299, 299))
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    img = tf.expand_dims(img, 0)
    return img


def preprocess_image(img, model_type):
    if model_type == ImageNetModelsPretrained.INCEPTION_V3.value:
        logging.info("preprocess image with inception model")

        img = tf.image.resize(img, (299, 299))
        img = tf.keras.applications.inception_v3.preprocess_input(img)
    else:
        logging.info("preprocess image with densenet model")

        img = tf.image.resize(img, (224, 224))
        img = tf.keras.applications.densenet.preprocess_input(img)

    img = tf.expand_dims(img, 0)
    return img


def get_inception_pretrained():
    image_model = tf.keras.applications.InceptionV3(include_top=False,  # because it is false, its doesnot have the last layer
                                                    weights='imagenet')
    new_input = image_model.input
    hidden_layer = image_model.layers[-1].output

    extractor_features_model = tf.keras.Model(new_input, hidden_layer)

    return extractor_features_model


class ImageNetModelsPretrained(Enum):
    INCEPTION_V3 = "inception"
    DENSENET = "densenet"


def get_extractor_features_model(model_type):
    if model_type == ImageNetModelsPretrained.INCEPTION_V3.value:
        logging.info("extract features with inception model")

        image_model = tf.keras.applications.InceptionV3(include_top=False,  # because it is false, its doesnot have the last layer
                                                        weights='imagenet')
    else:
        logging.info("extract features with densenet model")

        image_model = tf.keras.applications.densenet.DenseNet201(
            include_top=False, weights='imagenet')

    new_input = image_model.input
    hidden_layer = image_model.layers[-1].output

    extractor_features_model = tf.keras.Model(new_input, hidden_layer)

    return extractor_features_model


class FlipsAndRotations(Enum):
    FLIP_HORIZONTAL = 0
    FLIP_VERTICAL = 1
    FLIP_DIAGONAL = 2
    ROT_90 = 3
    ROT_180 = 4
    ROT_270 = 5
    ROT_360 = 6


def rotate_or_flip(img):
    mode = np.random.randint(len(FlipsAndRotations))

    if mode == FlipsAndRotations.FLIP_HORIZONTAL.value:
        new_img = tf.image.flip_left_right(img)
    elif mode == FlipsAndRotations.FLIP_VERTICAL.value:
        new_img = tf.image.flip_up_down(img)
    elif mode == FlipsAndRotations.FLIP_DIAGONAL.value:
        new_img = tf.image.flip_up_down(tf.image.flip_left_right(img))
    elif mode == FlipsAndRotations.ROT_90.value:
        new_img = tf.image.rot90(img, k=1)
    elif mode == FlipsAndRotations.ROT_180.value:
        new_img = tf.image.rot90(img, k=2)
    elif mode == FlipsAndRotations.ROT_270.value:
        new_img = tf.image.rot90(img, k=3)
    elif mode == FlipsAndRotations.ROT_360.value:
        new_img = tf.image.rot90(img, k=4)
    else:
        raise ValueError(
            "Mode should be equal to 0-6 (see ENUM FlipsAndRotations).")
    return new_img


# class Temperature(Enum):
#     WARM = 0
#     COLD = 1
#     NONE = 2


# def create_lut(x, y):
#     spl = UnivariateSpline(x, y)
#     return spl(range(65535))


# def rescale(value):
#     return round((value * 65535) / 255)


# def apply_lut(content, table, patch_sz):
#     prev_content, res = (content * 65535), []
#     for el in np.nditer(prev_content):
#         res.append(table[int(el)])
#     return (np.array(res) / 65535).reshape((patch_sz, patch_sz))


# def get_incr_lut():
#     global result_incr_lut
#     try:
#         return result_incr_lut
#     except NameError:
#         result_incr_lut = create_lut([0, rescale(64), rescale(128), rescale(192), rescale(256)],
#                                      [0, rescale(70), rescale(140), rescale(210), rescale(256)])
#         return dict(zip(list(range(len(result_incr_lut))), result_incr_lut))


# def get_decr_lut():
#     global result_decr_lut
#     try:
#         return result_decr_lut
#     except NameError:
#         result_decr_lut = create_lut([0, rescale(64), rescale(128), rescale(192), rescale(256)],
#                                      [0, rescale(30), rescale(80), rescale(120), rescale(192)])
#         return dict(zip(list(range(len(result_decr_lut))), result_decr_lut))


# def change_temperature(img):

#     patch_size = 128  # lado das imagens!!!

#     mode = np.random.randint(len(Temperature))

#     c_r, c_g, c_b = img[:, :, 0], img[:, :, 1], img[:, :, 2]

#     if mode == Temperature.WARM.value:
#         c_r = apply_lut(c_r, get_incr_lut(), patch_size)
#         c_b = apply_lut(c_b, get_decr_lut(), patch_size)

#     elif mode == Temperature.COLD.value:
#         c_r = apply_lut(c_r, get_decr_lut(), patch_size)
#         c_b = apply_lut(c_b, get_incr_lut(), patch_size)

#     elif mode == Temperature.NONE.value:
#         return img
#     else:
#         raise ValueError(
#             "Mode should be equal to 0 (warm), 1 (cold) or 3(None).")

#     return np.dstack((c_r, c_g, c_b, img[:, :, 3:]))
