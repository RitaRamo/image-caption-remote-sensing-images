# from models.abstract_model import AbstractModel
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
from models.attention_model import AttentionModel
from models.embeddings import get_glove_embeddings_matrix
from sklearn.metrics.pairwise import cosine_similarity
# This attention model has the first state of decoder all zeros (and not receiving the encoder states as initial state)
# Shape of the vector extracted from InceptionV3 is (64, 2048)


class Encoder(tf.keras.Model):

    def __init__(self, encoder_state_size, args):
        super(Encoder, self).__init__()

        self.dense = tf.keras.layers.Dense(
            encoder_state_size, activation="relu")

        self.args = args

    def call(self, input_images):

        if not self.args.fine_tuning:
            output = self.dense(input_images)

        else:
            finetuned_model = get_fine_tuning_model(self.args.image_model_type)

            images = finetuned_model(input_images)

            images = tf.reshape(
                images,
                (
                    images.shape[0],
                    -1,
                    images.shape[3]
                )
            )

            output = self.dense(images)

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
        # tf.nn.tanh(self.W1(features) + self.W2(hidden_with_time_axis)) shape == (batch_size, features_shape0, hidden_size/units)  [hidden_size==units]
        # score shape with self.V == (batch_size, features_shape0,1) ex: for inception (batch_size, 64, 1) -> because of Dense(1)]
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


class Decoder(tf.keras.Model):

    def __init__(self, embeddings_matrix, vocab_size, embedding_size, units):  # the state

        super(Decoder, self).__init__()

        self.attention = BahdanauAttention(units)
        # tf.keras.layers.Embedding(vocab_size, embedding_size, mask_zero=True)
        # TODO: keeping this embedding layer fixed and tied with pre-trained target output embeddings (Press
        # TENS Q POR AQUI Q É A DO GLOVE

        self.embedding = Embedding(vocab_size,
                                   embedding_size,
                                   mask_zero=True,
                                   weights=[embeddings_matrix
                                            ],
                                   trainable=False)
        # self.embedding = _get_embedding_layer(
        #     embedding_type, vocab_size, embedding_size, token_to_id)

        self.gru = tf.keras.layers.GRU(units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')
        self.dense = tf.keras.layers.Dense(
            vocab_size, activation="softmax")

        self.dense_continuous = tf.keras.layers.Dense(embedding_size)

        # todo:change -> activação linear ou sigmoid do mesmo tipo
        # self.embedding()

    def call(self, x, encoder_features, dec_hidden):
        # print("np shape x", np.shape(x))
        # print("np shape encoder_features", np.shape(encoder_features))
        # print("np shape dec_hidden", np.shape(dec_hidden))

        # defining attention as a separate model
        context_vector, attention_weights = self.attention(
            encoder_features, dec_hidden)

        # x shape == (batch_size,) (giving one word at a time)
        # x shape after passing through embedding == (batch_size, embedding_dim)
        # TODO: keeping this embedding layer fixed and tied with pre-trained target output embeddings (Press
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
        #output = self.dense(output)

        # TODO: ADD LAYER DENSE with size of embeddings Dense(emebddize_size)

        embedding_output = self.dense_continuous(output)
        # TODO: keeping this embedding layer fixed and tied with pre-trained target output embeddings (Press

        return embedding_output, dec_hidden, attention_weights, self.embedding  # , emebding_output


class AttentionContinuosModel(AttentionModel):

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
        self.embeddings_matrix = get_glove_embeddings_matrix(
            self.vocab_size, self.embedding_size, self.token_to_id)
        self.decoder = Decoder(self.embeddings_matrix,
                               self.vocab_size, self.embedding_size, self.units)

    def build(self):
        self.log_cosh = tf.keras.losses.LogCosh()
        self.optimizer = tf.keras.optimizers.Adam()

    def loss_function(self, real, embedding_predicted, embedding_layer):
        # real: y_true - one hot encoding shape [1, vocab size] -> get np.argmax before embbedging
        # pred: y_^ - probs of sotfmax [1, vocab size] -> get np.argmax before embbedging

        embedding_real = embedding_layer(
            np.argmax(real, axis=1))  # errado q deve ser o valor

        # convert what is not padding [!=0] to True, and padding [==0] to false
        mask = tf.math.logical_not(tf.math.equal(
            np.argmax(real, axis=1), self.token_to_id[PAD_TOKEN]))
        # mask = tf.math.logical_not(tf.math.equal(real, 0))

        loss_ = self.log_cosh(
            embedding_real,
            embedding_predicted
        )

        # converst True and Falses to 0s and 1s
        mask = tf.cast(mask, dtype=loss_.dtype)
        # loss is multiplied by masks values (1s and 0s), thus filtering the padding (0)
        loss_ *= mask

        # mean/avarage by batch_size

        return tf.reduce_mean(loss_)

    def calculate_tokens_loss(self, n_tokens,  dec_hidden, img_tensor, input_caption_seq,  target_caption_seq):
        loss = 0

        encoder_features = self.encoder(img_tensor)
        for i in range(n_tokens):

            predicted_output, dec_hidden, _, embedding_layer = self.decoder(
                input_caption_seq[:, i], encoder_features, dec_hidden)

            loss += self.loss_function(
                target_caption_seq[:, i], predicted_output, embedding_layer)

        return loss

    def train_step(self, img_tensor, input_caption_seq,  target_caption_seq):
        n_tokens = self.max_len-1
        dec_hidden = tf.zeros((self.args.batch_size, self.units))

        with tf.GradientTape() as tape:

            loss = self.calculate_tokens_loss(
                n_tokens, dec_hidden, img_tensor, input_caption_seq,  target_caption_seq)

        batch_loss = (loss / n_tokens)

        trainable_variables = self.encoder.trainable_variables + \
            self.decoder.trainable_variables

        gradients = tape.gradient(loss, trainable_variables)

        self.optimizer.apply_gradients(zip(gradients, trainable_variables))

        return batch_loss

    def validation_step(self, img_tensor, input_caption_seq,  target_caption_seq):
        # max_len -1 since the input_seq has max_len -1 (without end token)
        # and target_sentence is max_len -1 (without start token)
        n_tokens = self.max_len-1
        dec_hidden = tf.zeros((self.args.batch_size, self.units))

        loss = self.calculate_tokens_loss(
            n_tokens, dec_hidden, img_tensor, input_caption_seq,  target_caption_seq)

        batch_loss = (loss / n_tokens)

        return batch_loss

    def train(self, train_dataset, val_dataset, len_train_dataset, len_val_dataset):

        ckpt, ckpt_manager, start_epoch = self._load_latest_checkpoint()
        early_stop = EarlyStoppingWithCheckpoint(ckpt,
                                                 ckpt_manager,
                                                 baseline=ckpt.loss if start_epoch > 0 else None,
                                                 min_delta=0.0,
                                                 patience=10
                                                 )

        train_steps, val_steps = self._get_steps(
            len_train_dataset, len_val_dataset)

        for epoch in range(start_epoch, self.args.epochs):
            start = time.time()

            def calculate_total_loss(train_or_val, train_or_val_dataset, n_steps, train_or_val_step, epoch):
                total_loss = 0
                batch_i = 0
                for batch_samples in train_or_val_dataset.take(n_steps):

                    images_tensor = batch_samples[0]["input_1"]
                    input_caption_seq = batch_samples[0]["input_2"]
                    target_caption_seq = batch_samples[1]

                    batch_loss = train_or_val_step(images_tensor, input_caption_seq,
                                                   target_caption_seq)
                    total_loss += batch_loss

                    if batch_i % 5 == 0:
                        tf.print('{} -- Epoch {}/{}; Batch {}/{}; Loss {:.4f}'.format(
                            train_or_val, epoch, self.args.epochs, batch_i, n_steps, batch_loss))
                    batch_i += 1
                return total_loss

            # TRAIN
            total_loss = calculate_total_loss(
                "TRAIN", train_dataset, train_steps, self.train_step, epoch)

            epoch_loss = total_loss/train_steps

            tf.print('Time taken for 1 epoch {} sec'.format(
                time.time() - start))
            tf.print('\nTRAIN END! Epoch: {}; Loss: {:.4f}\n'.format(epoch,
                                                                     epoch_loss))  # N_BATCH -> n_steps

            # VALIDATION
            total_val_loss = calculate_total_loss(
                "VAL", val_dataset, val_steps, self.validation_step, epoch)

            epoch_val_loss = total_val_loss/val_steps

            early_stop.on_epoch_end(epoch, epoch_val_loss)

            tf.print('\n--------------- END EPOCH:{}⁄{}; Train Loss:{:.4f}; Val Loss:{:.4f} --------------\n'.format(
                epoch, self.args.epochs, epoch_loss, epoch_val_loss))  # N_BATCH -> n_steps

            if early_stop.is_to_stop_training():
                tf.print('Stop training. Best loss', ckpt.loss)
                break

    def generate_text(self, input_image):
        input_caption = np.zeros((1, self.max_len-1))

        decoder_sentence = START_TOKEN + " "

        input_caption = np.array([self.token_to_id[START_TOKEN]])

        i = 1

        dec_hidden = tf.zeros((1, self.units))

        while True:  # change to for!
            encoder_features = self.encoder(input_image)

            predicted_embedding_output, dec_hidden, attention_weights, embedding_layer = self.decoder(
                input_caption, encoder_features, dec_hidden)

            output_similarity_to_embeddings = cosine_similarity(
                predicted_embedding_output, self.embeddings_matrix)

            current_output_index = np.argmax(output_similarity_to_embeddings)

            #current_output_index = np.argmax(predicted_output)
            current_output_token = self.id_to_token[current_output_index]

            decoder_sentence += " " + current_output_token

            if (current_output_token == END_TOKEN or
                    i >= self.max_len-1):  # chegas ao 35
                break

            input_caption[0] = current_output_index

            i += 1

        print("decoded sentence", decoder_sentence)

        return decoder_sentence  # input_caption
