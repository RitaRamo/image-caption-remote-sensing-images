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
from optimizers.optimizers import get_optimizer
# This attention model has the first state of decoder all zeros (and not receiving the encoder states as initial state)
# Shape of the vector extracted from InceptionV3 is (64, 2048)

from models.attention_model import Encoder, BahdanauAttention, AttentionModel


class Decoder(tf.keras.Model):

    def __init__(self, embedding_type, vocab_size, embedding_size, token_to_id, units):  # the state

        super(Decoder, self).__init__()

        self.attention = BahdanauAttention(units)
        # tf.keras.layers.Embedding(vocab_size, embedding_size, mask_zero=True)
        self.embedding = _get_embedding_layer(
            embedding_type, vocab_size, embedding_size, token_to_id)

        self.lstm = tf.keras.layers.LSTM(units,
                                         return_sequences=True,
                                         return_state=True
                                         )
        self.dense = tf.keras.layers.Dense(
            vocab_size, activation="softmax")

    def call(self, x, encoder_features, dec_hidden):
        # print("np shape x", np.shape(x))
        # print("np shape encoder_features", np.shape(encoder_features))
        # print("np shape dec_hidden", np.shape(dec_hidden))

        # defining attention as a separate model
        context_vector, attention_weights = self.attention(
            encoder_features, dec_hidden)

        # x shape == (batch_size,) (giving one word at a time)
        # x shape after passing through embedding == (batch_size, embedding_dim)
        x = self.embedding(x)

        # tf.expand_dims(context_vector, 1) shape == (batch_size, 1,enc_embedding_dim) or hidden_size if enc_embedding == dec_unit
        # tf.expand_dims(x, 1) shape == (batch_size, 1, embedding_dim)
        # x shape after concatenation == (batch_size, 1,  enc_embedding_dim + embedding_dim) or hidden_size + embedding_dim if enc_embedding == dec_unit
        # ex: (batch_size, 1, 600) if hidden_size=300 and embding_dim is 300)
        x = tf.concat([tf.expand_dims(context_vector, 1),
                       tf.expand_dims(x, 1)], axis=-1)

        # passing the concatenated vector to the GRU
        # apply GRU output shape == (batch_size, 1, dec_units); dec_hidden shape == (batch_size, dec_units)
        output, dec_hidden, cell_state = self.lstm(x)

        # output shape == (batch_size, dec_units)
        output = tf.reshape(output, (-1, output.shape[2]))

        # output shape == (batch_size, vocab)
        output = self.dense(output)

        # TODO: ADD LAYER DENSE with size of embeddings Dense(emebddize_size)

        #emebding_output  = self.embedding(output)

        return output, dec_hidden, attention_weights  # , emebding_output

# duvidas o gru nao posmos initial state? e se fosse lstm? ->qual o hidden q davamos? dado ter 2 states? -> hidden!!!
# tenta fazer tu com o model de acordo cm a explicação
# qual e o initial state da gru?


class AttentionLSTMModel(AttentionModel):

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
        self.encoder = Encoder(self.embedding_size, self.args)
        self.decoder = Decoder(self.embedding_type,
                               self.vocab_size, self.embedding_size, self.token_to_id, self.units)
