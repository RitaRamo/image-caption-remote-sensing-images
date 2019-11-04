import os
import random
from pickle import dump, load

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from models.abstract_model import BATCH_SIZE
from models.simple_encoder_decoder import SimpleEncoderDecoderModel
from models.simple_model import SimpleModel
from preprocess_data.augmented_data.preprocess_augmented_data import (
    augmented_generator, get_images_loaded)
from preprocess_data.simple_data.preprocess_data import (extract_features,
                                                         simple_generator)
from preprocess_data.tokens import (END_TOKEN, START_TOKEN,
                                    convert_captions_to_Y, preprocess_tokens)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['PYTHONHASHSEED'] = '0'

os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)

FEATURE_EXTRACTION = True
AUGMENT_DATA = False

PATH = "datasets/RSICD/"
IMAGE_PATH = "/RSICD_images/"


def get_images_and_captions(dataset):
    images_names = []
    captions_of_tokens = []
    for row in dataset["images"]:
        image_name = row["filename"]
        for caption in row["sentences"]:
            tokens = [START_TOKEN] + caption["tokens"] + [END_TOKEN]

            captions_of_tokens.append(tokens)
            images_names.append(image_name)

    images_names, captions_of_tokens = shuffle(
        images_names, captions_of_tokens, random_state=42)
    return images_names, captions_of_tokens


# run as: PYTHONHASHSEED=0 python train.py
if __name__ == "__main__":
    print("load and split dataset")
    raw_dataset = pd.read_json(PATH+"dataset_rsicd.json")
    raw_dataset = raw_dataset.sample(frac=1, random_state=42)
    train, validation, test = np.split(
        raw_dataset, [int(.8*len(raw_dataset)), int(.9*len(raw_dataset))])

    print(
        "transform dataset into respective images [[img_name]...] and captions [[token1,token2,...]...]")
    train_images_names, train_captions_of_tokens = get_images_and_captions(
        train)
    val_images_names, val_captions_of_tokens = get_images_and_captions(
        validation)
    test_images_names, test_captions_of_tokens = get_images_and_captions(test)

    vocab_size, token_to_id, id_to_token, max_len = preprocess_tokens(
        train_captions_of_tokens
    )  # preprocess should be done with trainset

    print("captions tokens -> captions ids, since the NN doesnot read tokens but rather numbers")
    train_input_captions, train_target_captions = convert_captions_to_Y(
        train_captions_of_tokens, max_len, token_to_id)
    val_input_captions, val_target_captions = convert_captions_to_Y(
        val_captions_of_tokens, max_len, token_to_id)
    test_input_captions, test_target_captions = convert_captions_to_Y(
        test_captions_of_tokens, max_len, token_to_id)

    print("images names -> images vectors (respective representantion of an image)")
    train_generator = None
    val_generator = None

    images_features = extract_features(
        [row["filename"] for row in raw_dataset["images"]],
        file_of_dumped_features="./preprocess_data/simple_data/all_image_features.pkl"
    )

    if FEATURE_EXTRACTION:

        if AUGMENT_DATA:
            print("Augmented images")

            dict_name_img = get_images_loaded(
                [row["filename"] for row in raw_dataset["images"]],
                file_of_dumped_features="/Users/RitaRamos/Documents/INESC-ID/remote-sensing/notebooks/loaded_image_features.pkl")

            train_generator = augmented_generator(
                dict_name_img,
                train_images_names,
                train_input_captions,
                train_target_captions,
                vocab_size
            )

            val_generator = augmented_generator(
                dict_name_img,
                val_images_names,
                val_input_captions,
                val_target_captions,
                vocab_size
            )

        else:
            print("Not augmented images")

            train_generator = simple_generator(
                images_features,
                train_images_names,
                train_input_captions,
                train_target_captions,
                vocab_size
            )

            val_generator = simple_generator(
                images_features,
                val_images_names,
                val_input_captions,
                val_target_captions,
                vocab_size
            )

        print("create generators for datasets (train and val)")
        train_dataset = tf.data.Dataset.from_generator(
            lambda: train_generator,
            ({'input_1': tf.float32, 'input_2': tf.float32}, tf.float32)
        ).batch(BATCH_SIZE)

        val_dataset = tf.data.Dataset.from_generator(
            lambda: val_generator,
            ({'input_1': tf.float32, 'input_2': tf.float32}, tf.float32)
        ).batch(BATCH_SIZE)

        print("create and run model")
        model = SimpleModel(
            vocab_size,
            max_len,
            131072
        )  # SimpleEncoderDecoderModel

        model.create()
        model.build()
        model.train(train_dataset, val_dataset,
                    BATCH_SIZE, len(val_images_names))
        model.save()

        # img_name = train_images_names[0]
        # its_caption = train_images_names[0]
        # print("any image", img_name)
        # # print("pois esta é a caption", its_caption)

        # features = images_features[img_name]
        # img_tensor = tf.reshape(features, [-1])
        # print("img_tensor shape", np.shape(img_tensor))
        # print("with expa", tf.expand_dims(
        #     img_tensor, axis=0))

        # print("vamos tentar o new model", model.generate_text(
        #     tf.expand_dims(
        #         img_tensor, axis=0), token_to_id, id_to_token))

    else:  # pre-trained model, thus only preprocc images
        print("Not implemented yet pretrainable")
