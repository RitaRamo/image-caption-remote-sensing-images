# from models.abstract_model import AbstractModel
from models.abstract_model import AbstractModel
from models.layers import _get_embedding_layer
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding, TimeDistributed
from tensorflow.keras.models import Model
import numpy as np
import tensorflow as tf
import time
from preprocess_data.tokens import START_TOKEN, END_TOKEN, PAD_TOKEN
from preprocess_data.images import get_fine_tuning_model
import logging
from models.callbacks import EarlyStoppingWithCheckpoint
# This attention model has the first state of decoder all zeros (and not receiving the encoder states as initial state)
# Shape of the vector extracted from InceptionV3 is (64, 2048)
from models.attention_model import AttentionModel, Encoder, Decoder


class AttentionEncInitialStateModel(AttentionModel):

    MODEL_DIRECTORY = "././experiments/results/AttentionModel/"

    def __init__(
        self,
        args,
        vocab_size,
        max_len,
        token_to_id,
        id_to_token,
        encoder_input_size,
        embedding_type=None,
        units=256
    ):
        super().__init__(args, vocab_size, max_len, token_to_id, id_to_token,
                         encoder_input_size, embedding_type, units)
        self.model = None

    def create(self):
        self.encoder = Encoder(self.units, self.args)
        self.decoder = Decoder(self.embedding_type,
                               self.vocab_size, self.embedding_size, self.token_to_id, self.units)

    def calculate_tokens_loss(self, n_tokens,  dec_hidden, img_tensor, input_caption_seq,  target_caption_seq):
        loss = 0

        encoder_features = self.encoder(img_tensor)

        hidden = tf.math.reduce_mean(encoder_features, axis=1)

        for i in range(n_tokens):

            predicted_output, hidden, _ = self.decoder(
                input_caption_seq[:, i], encoder_features, hidden)

            loss += self.loss_function(
                target_caption_seq[:, i], predicted_output)

        return loss
