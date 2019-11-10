import os
import numpy as np
from collections import OrderedDict
from toolz.itertoolz import unique

START_TOKEN = "<start_seq>"
END_TOKEN = "<end_seq>"
PAD_TOKEN = "#"
OOV_TOKEN = "<unk>"
os.environ['PYTHONHASHSEED'] = '0'


def preprocess_tokens(train_captions):
    all_tokens = [START_TOKEN, END_TOKEN, OOV_TOKEN]
    for caption_tokens in train_captions:
        all_tokens.extend(caption_tokens)

    #vocab = list(set(all_tokens))
    vocab = list(unique(all_tokens))
    token_to_id = OrderedDict([(value, index+1)
                               for index, value in enumerate(vocab)])
    id_to_token = OrderedDict([(index+1, value)
                               for index, value in enumerate(vocab)])

    token_to_id[PAD_TOKEN] = 0
    id_to_token[0] = PAD_TOKEN

    len_vocab = len(vocab) + 1  # padding token

    max_len = max(map(len, train_captions))

    return len_vocab, token_to_id, id_to_token, max_len


def convert_captions_to_Y(captions_of_tokens, max_len, token_to_id):
    input_captions = np.zeros(
        (len(captions_of_tokens), max_len-1), dtype='f') + token_to_id[PAD_TOKEN]
    target_captions = np.zeros(
        (len(captions_of_tokens), max_len-1), dtype='f') + token_to_id[PAD_TOKEN]

    for i in range(len(captions_of_tokens)):
        tokens_to_integer = [token_to_id.get(
            token, token_to_id[OOV_TOKEN]) for token in captions_of_tokens[i]]
        input_caption = tokens_to_integer[:max_len-1]
        target_caption = tokens_to_integer[1:][:max_len-1]

        input_captions[i, :len(input_caption)] = input_caption
        target_captions[i, :len(target_caption)] = target_caption
    return input_captions, target_captions
