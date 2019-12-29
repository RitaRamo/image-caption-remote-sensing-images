# from models.abstract_model import AbstractModel
from models.abstract_model import AbstractModel

from tensorflow.keras.layers import Input, LSTM, Dense, Embedding, TimeDistributed
from tensorflow.keras.models import Model
import numpy as np
import tensorflow as tf
import time
from preprocess_data.tokens import START_TOKEN, END_TOKEN

# This attention model has the first state of decoder all zeros (and not receiving the encoder states as initial state)
# Shape of the vector extracted from InceptionV3 is (64, 2048)


class Encoder(tf.keras.Model):

    def __init__(self, encoder_state_size):
        super(Encoder, self).__init__()

        self.dense = tf.keras.layers.Dense(
            encoder_state_size, activation="relu")

    def call(self, images_features):
        output = self.dense(images_features)

        # if fine_tuning
        # else
        return output


class BahdanauAttention(tf.keras.Model):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, features, hidden):
        # features(CNN_encoder output) shape == (batch_size, features_shape0, embedding_dim)
        # ex: for inception (batch_size, 64, embedding_dim)

        # hidden shape == (batch_size, hidden_size/dec_units)
        # hidden_with_time_axis shape == (batch_size, 1, hidden_size/dec_units)
        hidden_with_time_axis = tf.expand_dims(hidden, 1)

        # self.W1(features) shape ==  (batch_size, features_shape0, hidden_size/units)
        # self.W2(hidden_with_time_axis) shape ==  (batch_size, 1, hidden_size/units)
        # tf.nn.tanh(self.W1(features) + self.W2(hidden_with_time_axis)) shape == (batch_size, features_shape0, hidden_size)  [hidden_size==units]
        # score shape with self.V == (batch_size, features_shape0,1) ex: for inception (batch_size, 64, 1)
        score = self.V(tf.nn.tanh(self.W1(features) +
                                  self.W2(hidden_with_time_axis)))

        # attention_weights shape == (batch_size, features_shape0, 1) ex: for inception (batch_size, 64, 1)
        # you get 1 at the last axis because you are applying score to self.V (softmax to each features shapes)
        attention_weights = tf.nn.softmax(score, axis=1)

        # context_vector shape == (batch_size, feature_shape0, features_shape1[enc_embedding])
        # it could be shape == (batch_size, feature_shape0, hidden_size, if enc_embedding == dec_unit)
        context_vector = attention_weights * features

        # context_vector shape after sum == (batch_size, features_shape1/enc_embedding)
        # it could be shape == (batch_size,  hidden_size) if enc_embedding == hidden_size/dec_unit)
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights

# if
#encoder = Encoder( units)
#decoder = Decoder(vocab_size, embedding_dim, units)


class Decoder(tf.keras.Model):

    def __init__(self, vocab_size, embedding_size, units):  # the state

        super(Decoder, self).__init__()

        self.attention = BahdanauAttention(units)
        self.embedding = tf.keras.layers.Embedding(
            vocab_size, embedding_size, mask_zero=True)
        self.gru = tf.keras.layers.GRU(units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')
        self.dense = tf.keras.layers.Dense(
            vocab_size, activation="softmax")

    def call(self, x, encoder_features, dec_hidden):

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
        output, dec_hidden = self.gru(x)

        # output shape == (batch_size, dec_units)
        output = tf.reshape(output, (-1, output.shape[2]))

        # output shape == (batch_size, vocab)
        output = self.dense(output)

        return output, dec_hidden, attention_weights

# duvidas o gru nao posmos initial state? e se fosse lstm? ->qual o hidden q davamos? dado ter 2 states?
# tenta fazer tu com o model de acordo cm a explicação
# qual e o initial state da gru?


class AttentionModel(AbstractModel):

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
        embedding_size=300,
        units=256
    ):
        super().__init__(args, vocab_size, max_len, token_to_id, id_to_token,
                         encoder_input_size, embedding_type, embedding_size, units)
        self.model = None

    def create(self):
        self.encoder = Encoder(self.embedding_size)
        self.decoder = Decoder(
            self.vocab_size, self.embedding_size, self.units)

    def build(self):
        self.crossentropy = tf.keras.losses.CategoricalCrossentropy()
        self.optimizer = tf.keras.optimizers.Adam()

    def summary(self):
        pass

    def save(self):
        pass
        # self.model.save(self.get_path())

    def train_step(self, img_tensor, input_caption_seq,  target_caption_seq):

        loss = 0
        dec_hidden = tf.zeros((self.args.batch_size, self.units))

        # max_len -1 since the input_seq has max_len -1 (without end token)
        # and target_sentence is max_len -1 (without start token)
        n_tokens = self.max_len-1

        with tf.GradientTape() as tape:

            encoder_features = self.encoder(img_tensor)

            for i in range(n_tokens):

                predicted_output, dec_hidden, attention_weights = self.decoder(
                    input_caption_seq[:, i], encoder_features, dec_hidden)
                loss += self.crossentropy(
                    target_caption_seq[:, i], predicted_output)

        batch_loss = (loss / n_tokens)

        trainable_variables = self.encoder.trainable_variables + \
            self.decoder.trainable_variables

        gradients = tape.gradient(loss, trainable_variables)

        self.optimizer.apply_gradients(zip(gradients, trainable_variables))

        return batch_loss

    def train(self, train_dataset, val_dataset, len_train_dataset, len_val_dataset):

        if self.args.disable_steps:
            train_steps = 1
            val_steps = 1
        else:
            train_steps = int(len_train_dataset/self.args.batch_size)
            val_steps = int(len_val_dataset/self.args.batch_size)

        for epoch in range(self.args.epochs):
            start = time.time()
            total_loss = 0

            # n_batch ->put from tqdm import tqdm
            i = 0
            for batch_samples in train_dataset.take(train_steps):

                images_tensor = batch_samples[0]["input_1"]
                input_caption_seq = batch_samples[0]["input_2"]
                target_caption_seq = batch_samples[1]

                batch_loss = self.train_step(images_tensor, input_caption_seq,
                                             target_caption_seq)
                total_loss += batch_loss

                # if batch % 100 == 0:
                # cenas
                tf.print('Epoch {} Batch {} Loss {:.4f}'.format(
                    epoch, i, batch_loss))
                i += 1

            epoch_loss = total_loss/train_steps
            tf.print('Epoch {} Loss {:.6f}'.format(epoch,
                                                   total_loss/train_steps))  # N_BATCH -> n_steps
            tf.print('Time taken for 1 epoch {} sec\n'.format(
                time.time() - start))

            # TODO: do same logic but for validation!!
            # TODO: try to do early stop
