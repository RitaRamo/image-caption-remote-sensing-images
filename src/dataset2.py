# até ao 12:00 acabar isto!!
import json
import logging
import nltk
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from collections import defaultdict
from generators.abstract_generator import PATH
from preprocess_data.tokens import (END_TOKEN, START_TOKEN,
                                    convert_captions_to_Y, preprocess_tokens)
import re


def _get_images_and_captions(dataset):
    images_names = []
    captions_of_tokens = []
    for row in dataset["images"]:
        image_name = row["filename"]
        for caption in row["sentences"]:
            caption = re.sub(r'\.', r'', caption["raw"])

            tokens = [START_TOKEN] + \
                tokenizer.tokenize(caption) + [END_TOKEN]

            captions_of_tokens.append(tokens)
            images_names.append(image_name)

    images_names, captions_of_tokens = shuffle(
        images_names, captions_of_tokens, random_state=42)
    return images_names, captions_of_tokens


def _get_dict_image_and_its_captions(dataset):
    images_captions = defaultdict(list)
    for row in dataset["images"]:
        image_name = row["filename"]
        for caption in row["sentences"]:
            caption = re.sub(r'\.', r'', caption["raw"])
            caption_of_tokens = START_TOKEN + " " + caption + END_TOKEN

            images_captions[image_name].append(caption_of_tokens)

    return images_captions


def _dump_dict_to_json(dict, file_dir, file_name):
    with open(file_dir+file_name, 'w+') as f:
        json.dump(dict, f, indent=2)


def _dump_data_to_json(images_names, captions_tokens, file_dir, file_name):
    dataset_dict = {
        "images_names": images_names,
        "captions_tokens": captions_tokens
    }
    # falta directori
    _dump_dict_to_json(dataset_dict, file_dir, file_name)
    # with open(file_dir+file_name, 'w+') as f:
    #     json.dump(dataset_dict, f, indent=2)


def _dump_vocab_to_json(vocab_size, token_to_id, id_to_token, max_len, file_dir):
    vocab_info = {}
    vocab_info["vocab_size"] = vocab_size
    vocab_info["token_to_id"] = token_to_id
    vocab_info["id_to_token"] = id_to_token
    vocab_info["max_len"] = max_len

    _dump_dict_to_json(vocab_info, file_dir, "vocab_info.json")

    # with open(file_dir+"vocab_info.json", 'w+') as f:
    #     json.dump(vocab_info, f, indent=2)


def _save_dataset(raw_dataset, file_dir):
    # suffle and split dataset into train,val and test
    raw_dataset = raw_dataset.sample(frac=1, random_state=42)
    train, validation, test = np.split(
        raw_dataset, [int(.8*len(raw_dataset)), int(.9*len(raw_dataset))])

    # "transform dataset into respective images [[img_name]...] and captions [[token1,token2,...]...]")
    train_images_names, train_captions_of_tokens = _get_images_and_captions(
        train)
    val_images_names, val_captions_of_tokens = _get_images_and_captions(
        validation)

    test_dict_image_captions = _get_dict_image_and_its_captions(test)

    vocab_size, token_to_id, id_to_token, max_len = preprocess_tokens(
        train_captions_of_tokens
    )  # preprocess should be done with trainset

    # save vocab and datasets
    _dump_vocab_to_json(vocab_size, token_to_id,
                        id_to_token, max_len, file_dir)

    _dump_data_to_json(train_images_names, train_captions_of_tokens,
                       file_dir, "train.json")
    _dump_data_to_json(val_images_names, val_captions_of_tokens,
                       file_dir, "val.json")

    _dump_dict_to_json(test_dict_image_captions, file_dir, "test.json")


def get_dataset(file_path):
    with open(file_path) as json_file:
        dataset = json.load(json_file)
    return dataset


def get_vocab_info(file_dir):
    with open(file_dir+"vocab_info.json") as json_file:
        vocab_info = json.load(json_file)

    # given it was loaded from a json, the dict id_to_token has keys as strings instead of int, as supposed. To fix:
    vocab_info["id_to_token"] = {
        int(k): v for k, v in vocab_info["id_to_token"].items()}

    return vocab_info


if __name__ == "__main__":
    logging.basicConfig(
        format='%(levelname)s: %(message)s', level=logging.INFO)

    logging.info("start to save datasets and vocab of of RSCID")
    nltk.download('wordnet')
    tokenizer = nltk.tokenize.WordPunctTokenizer()

    raw_dataset = pd.read_json(PATH + "dataset_rsicd.json")
    _save_dataset(raw_dataset, "src/datasets/RSICD/dataset2/")

    logging.info("saved datasets and vocab")
