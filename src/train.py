from __future__ import absolute_import, division, print_function, unicode_literals

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
from models.simple_model_finetuning import SimpleFineTunedModel
from preprocess_data.augmented_data.preprocess_augmented_data import (
    augmented_generator, get_images_loaded)
from preprocess_data.simple_data.preprocess_data import (extract_features,
                                                         simple_generator)
from preprocess_data.tokens import (END_TOKEN, START_TOKEN,
                                    convert_captions_to_Y, preprocess_tokens)

from generators.features_extracted.simple_generator import FeaturesExtractedSimpleGenerator
from generators.features_extracted.augmented_generator import FeaturesExtractedAugmentedGenerator
from generators.fine_tuned.simple_generator import FineTunedSimpleGenerator
from generators.features_extracted.augmented_generator import FineTunedAugmentedGenerator
import logging
import argparse

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['PYTHONHASHSEED'] = '0'

os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)

FEATURE_EXTRACTION = True
AUGMENT_DATA = False

PATH = "src/datasets/RSICD/"
IMAGE_PATH = "/RSICD_images/"


parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
parser.add_argument(
    '--file_name', help='name of file that was used to fill the all other arguments', default=None)
# parser.add_argument('--fine_tuning',
#                     help='set True to use feature extraction, otherwise fine-tuning', default=False, type=bool)
# parser.add_argument(
#     '--augment_data', help='set True to augment data', default=False)

parser.add_argument(
    '--model_class_str', help='class name of the model to train', default="SimpleModel")

parser.add_argument('--augment_data', action='store_true', default=False,
                    help='Set a switch to true')

parser.add_argument('--fine_tuning', action='store_true', default=False,
                    help='Set a switch to true')


args = parser.parse_args()

print(args.__dict__)


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
    logging.basicConfig(
        format='%(levelname)s: %(message)s', level=logging.INFO)

    logging.info("Num GPUs Available: %s", len(
        tf.config.experimental.list_physical_devices('GPU')))

    logging.info("load and split dataset")
    raw_dataset = pd.read_json(PATH+"dataset_rsicd.json")
    raw_dataset = raw_dataset.sample(frac=1, random_state=42)
    train, validation, test = np.split(
        raw_dataset, [int(.8*len(raw_dataset)), int(.9*len(raw_dataset))])

    logging.info(
        "transform dataset into respective images [[img_name]...] and captions [[token1,token2,...]...]")
    train_images_names, train_captions_of_tokens = get_images_and_captions(
        train)
    val_images_names, val_captions_of_tokens = get_images_and_captions(
        validation)
    test_images_names, test_captions_of_tokens = get_images_and_captions(test)

    vocab_size, token_to_id, id_to_token, max_len = preprocess_tokens(
        train_captions_of_tokens
    )  # preprocess should be done with trainset

    vocab_info = {}
    vocab_info["vocab_size"] = vocab_size
    vocab_info["token_to_id"] = token_to_id
    vocab_info["id_to_token"] = id_to_token
    vocab_info["max_len"] = max_len

    logging.info(
        "captions tokens -> captions ids, since the NN doesnot read tokens but rather numbers")
    train_input_captions, train_target_captions = convert_captions_to_Y(
        train_captions_of_tokens, max_len, token_to_id)
    val_input_captions, val_target_captions = convert_captions_to_Y(
        val_captions_of_tokens, max_len, token_to_id)
    test_input_captions, test_target_captions = convert_captions_to_Y(
        test_captions_of_tokens, max_len, token_to_id)

    logging.info(
        "images names -> images vectors (respective representantion of an image)")
    train_generator = None
    val_generator = None

    # images_features = extract_features(
    #     [row["filename"] for row in raw_dataset["images"]],
    #     file_of_dumped_features="/Users/RitaRamos/Documents/INESC-ID/image-caption-remote-sensing-images/generators/all_image_features.pkl"
    # )

    model = None

    print("my args again", args.__dict__)
    print("my fine_tuning", args.fine_tuning)

    if args.fine_tuning:
        logging.info("Fine Tuning")

        if args.augment_data:
            logging.info("with augmented images")
            generator = FineTunedAugmentedGenerator(raw_dataset)

        else:
            logging.info("without augmented images")
            generator = FineTunedSimpleGenerator(raw_dataset)

    else:
        logging.info("Feature extraction")

        if args.augment_data:
            logging.info("with augmented images")

            generator = FeaturesExtractedAugmentedGenerator(raw_dataset)

        else:
            logging.info("without augmented images")

            generator = FeaturesExtractedSimpleGenerator(raw_dataset)

    logging.info("create generators for datasets (train and val)")

    train_generator = generator.generate(
        train_images_names,
        train_input_captions,
        train_target_captions,
        vocab_size
    )

    val_generator = generator.generate(
        val_images_names,
        val_input_captions,
        val_target_captions,
        vocab_size
    )

    train_dataset = tf.data.Dataset.from_generator(
        lambda: train_generator,
        ({'input_1': tf.float32, 'input_2': tf.float32}, tf.float32)
    ).batch(BATCH_SIZE)

    val_dataset = tf.data.Dataset.from_generator(
        lambda: val_generator,
        ({'input_1': tf.float32, 'input_2': tf.float32}, tf.float32)
    ).batch(BATCH_SIZE)

    logging.info("create and run model")
    logging.info("qual é o tamanho do input %s",
                 generator.get_shape_of_input_image())
    # model = SimpleModel(
    #     str(args.__dict__),
    #     vocab_size,
    #     max_len,
    #     generator.get_shape_of_input_image()
    # )

    #myvar = SimpleModel

    model_class = globals()[args.model_class_str]

    model = model_class(
        str(args.__dict__),
        vocab_size,
        max_len,
        generator.get_shape_of_input_image()
    )

    # SimpleEncoderDecoderModel

    model.create()
    model.summary()
    model.build()
    model.train(train_dataset, val_dataset,
                BATCH_SIZE, len(val_images_names))
    model.save()

    img_name = train_images_names[0]
    its_caption = train_images_names[0]
    logging.info("any image %s", img_name)
    # logging.info("pois esta é a caption", its_caption)

    # features = images_features[img_name]
    # img_tensor = tf.reshape(features, [-1])
    # logging.info("img_tensor shape %s", np.shape(img_tensor))
    # logging.info("with expa %s", tf.expand_dims(
    #     img_tensor, axis=0))

    # logging.info("vamos tentar o new model %s", model.generate_text(
    #     tf.expand_dims(
    #         img_tensor, axis=0), token_to_id, id_to_token))

# Hoje
# dar erro qd o modelo q tens de usar qd tas em feature extraction
if feature:
    class_name in []

# perceber como é feito o Test!! (eu penso q é fazer load tal como aqui e depois testar)
# 1 º tens q ter algo q faz save do dataset! (n em cenas -> train_images, train_captions...)
Rscid:
    raw{

    }
    dataset
    train
    val
    test
dataset.py

# 2 º tens q ter vocab_load


# Save token to id, max_len,len para os testes usarem!![de manha]
# Tentar por teste e assim (a idea gerar de loading) -> save do modelo e se é fine-tuning ou não [até às 17:00]
# por os modelos numa configuração (qual o modelo q vais usar...) [fazer os files]

# por os modelos a correr o mesmo, fazer last check com todos!!
# Por os valores dos modelos correctos (epochs, etc)
# por os modelos a correr o mesmo , fazer last check com todos!!
# Por a correr online (scp) [GOAL:hoje!!!]

# Por a outra LSTM (até as 16:00)
# Noite: Por embeddings!! (até as 15:00)
# por as métricas dos modelos

# Amanhã:
# Focar no test set, se é smp o mesmo!!

# Amanhã:
# 3 - Tenta usar word embeddings pre-treinados, por exemplo os do projecto GloVe ( https://keras.io/examples/pretrained_word_embeddings/ )

# Quarta:
# 2 - Troca também a LSTM por uma MultiplicativeLSTM ( https://github.com/titu1994/Keras-Multiplicative-LSTM ).

# 4 - Começar a fazer uma loss function diferente, que use outputs continuos como nos trabalhos da Yulia e do Graham de que te falei. Uma sugestão nesse sentido passa pela seguinte alteração sobre o modelo que tens agora.


# Por o outro modelo de Encoder vs Decoder a funcar!!
# por save do modelo (usar checkpoint!!)
# por geração!
# ver se consegues por fit_genrator, ao ter o Model(input=encoder, outputs=decoder)
# por mask!
# por com fine-tuning!
