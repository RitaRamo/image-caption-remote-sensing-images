from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import logging
import os
import random
from pickle import dump, load
import tensorflow as tf
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from args_parser import get_args
from dataset import get_dataset, get_vocab_info
from generators.abstract_generator import PATH
from generators.features_extracted.augmented_generator import (
    FeaturesExtractedAugmentedGenerator, FineTunedAugmentedGenerator)
from generators.features_extracted.simple_generator import \
    FeaturesExtractedSimpleGenerator
from generators.fine_tuned.simple_generator import FineTunedSimpleGenerator
from models.simple_encoder_decoder import SimpleEncoderDecoderModel
from models.simple_model import SimpleModel
from models.simple_model2 import SimpleModel2
from models.attention_model import AttentionModel
from models.attention_model1 import AttentionModel1


from models.simple_model_finetuning import SimpleFineTunedModel


from preprocess_data.tokens import (END_TOKEN, START_TOKEN,
                                    convert_captions_to_Y, preprocess_tokens)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['PYTHONHASHSEED'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)

FEATURE_EXTRACTION = True
AUGMENT_DATA = False


# run as: PYTHONHASHSEED=0 python3 src/train.py @experiments/conf_files/FE_S_SM.txt
if __name__ == "__main__":
    logging.basicConfig(
        format='%(levelname)s: %(message)s', level=logging.INFO)

    logging.info("tensorflow version %s", tf.__version__)

    args = get_args()
    logging.info(args.__dict__)

    logging.info("Num GPUs Available: %s", len(
        tf.config.experimental.list_physical_devices('GPU')))

    print("gpu is available", tf.test.is_gpu_available())

    logging.info("load and split dataset")
    raw_dataset = pd.read_json(PATH + "dataset_rsicd.json")

    vocab_info = get_vocab_info("src/datasets/RSICD/dataset/")
    vocab_size, token_to_id, id_to_token, max_len = vocab_info[
        "vocab_size"], vocab_info["token_to_id"], vocab_info["id_to_token"], vocab_info["max_len"]

    train = get_dataset(
        "src/datasets/RSICD/dataset/train.json")
    train_images_names, train_captions_of_tokens = train["images_names"], train["captions_tokens"]
    logging.info("len train_images_names %s", len(train_images_names))

    val = get_dataset(
        "src/datasets/RSICD/dataset/val.json")
    val_images_names, val_captions_of_tokens = val["images_names"], val["captions_tokens"]
    logging.info("len val_images_names %s", len(val_images_names))

    logging.info(
        "captions tokens -> captions ids, since the NN doesnot read tokens but rather numbers")
    train_input_captions, train_target_captions = convert_captions_to_Y(
        train_captions_of_tokens, max_len, token_to_id)
    val_input_captions, val_target_captions = convert_captions_to_Y(
        val_captions_of_tokens, max_len, token_to_id)

    logging.info(
        "images names -> images vectors (respective representantion of an image)")
    train_generator = None
    val_generator = None

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
    ).batch(args.batch_size)

    val_dataset = tf.data.Dataset.from_generator(
        lambda: val_generator,
        ({'input_1': tf.float32, 'input_2': tf.float32}, tf.float32)
    ).batch(args.batch_size)

    # for i in train_dataset.take(2):
    #     print("opaaa", np.shape(i))
    #     print("ai", np.shape(i[0]))
    #     print("a2", np.shape(i[1]))
    # print("agora é q e", next(iter(train_dataset)))

    # def gen():
    #     lista = [0, 1, 2, 3, 4, 5]
    #     while True:
    #         for i in lista:
    #             yield i

    # train_dataset = tf.data.Dataset.from_generator(
    #     gen,
    #     tf.int64
    # ).batch(args.batch_size)

    # # train_dataset = iter(train_dataset)
    # # for a in range(3):  # [0,1,2,3] ; [4,5,0,1];[2,3,4,5]
    # #     # for i in train_dataset.take(2): [1,2]; [3,4]
    # #     #     print("opaaa", i)
    # #     print("opaa memso", a, next((train_dataset)))

    # # for a in range(5):
    # for i in train_dataset.take(5):
    #     print("\ola", i)

    logging.info("create and run model")
    logging.info("qual é o tamanho do input %s",
                 generator.get_shape_of_input_image())

    model_class = globals()[args.model_class_str]

    model = model_class(
        args,
        vocab_size,
        max_len,
        token_to_id,
        id_to_token,
        generator.get_shape_of_input_image(),
        args.embedding_type
    )

    model.create()
    model.summary()
    model.build()
    model.train(train_dataset, val_dataset,
                len(train_images_names), len(val_images_names))

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


# Por a correr online (scp) [GOAL:hoje!!!]
# Por a outra LSTM (até as 16:00)
# (yeah por embeddings)
# tratar do erro dos imports!
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
