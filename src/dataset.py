# até ao 12:00 acabar isto!!
import json
import logging

import numpy as np
import pandas as pd
from sklearn.utils import shuffle

from generators.abstract_generator import PATH
from preprocess_data.tokens import (END_TOKEN, START_TOKEN,
                                    convert_captions_to_Y, preprocess_tokens)


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


def dump_data_to_json(images_names, captions_tokens, file_dir, file_name):
    dataset_dict = {
        "images_names": images_names,
        "captions_tokens": captions_tokens
    }
    # falta directori
    with open(file_dir+file_name, 'w+') as f:
        json.dump(dataset_dict, f, indent=2)


def dump_vocab_to_json(vocab_size, token_to_id, id_to_token, max_len, file_dir):
    vocab_info = {}
    vocab_info["vocab_size"] = vocab_size
    vocab_info["token_to_id"] = token_to_id
    vocab_info["id_to_token"] = id_to_token
    vocab_info["max_len"] = max_len

    with open(file_dir+"vocab_info.json", 'w+') as f:
        json.dump(vocab_info, f, indent=2)


def save_dataset(raw_dataset, file_dir):
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

    dump_vocab_to_json(vocab_size, token_to_id, id_to_token, max_len, file_dir)

    dump_data_to_json(train_images_names, train_captions_of_tokens,
                      file_dir, "train.json")
    dump_data_to_json(val_images_names, val_captions_of_tokens,
                      file_dir, "val.json")
    dump_data_to_json(test_images_names, test_captions_of_tokens,
                      file_dir,  "test.json")


def get_dataset(file_path):
    with open(file_path) as json_file:
        vocab_info = json.load(json_file)
    return vocab_info


def get_vocab_info(file_dir):
    with open(file_dir+"vocab_info.json") as json_file:
        vocab_info = json.load(json_file)
    return vocab_info


if __name__ == "__main__":
    # save train,test and val of RSCID
    raw_dataset = pd.read_json(PATH + "dataset_rsicd.json")
    save_dataset(raw_dataset, "src/datasets/RSICD/dataset/")