# from models.abstract_model import AbstractModel
from models.abstract_model import AbstractModel

from tensorflow.keras.layers import Input, LSTM, Dense, Embedding, TimeDistributed
from tensorflow.keras.models import Model
import numpy as np
import tensorflow as tf
import time
from preprocess_data.tokens import START_TOKEN, END_TOKEN, PAD_TOKEN

# This attention model has the first state of decoder all zeros (and not receiving the encoder states as initial state)
# Shape of the vector extracted from InceptionV3 is (64, 2048)


class Encoder(tf.keras.Model):

    def __init__(self, encoder_state_size):
        super(Encoder, self).__init__()

        self.dense = tf.keras.layers.Dense(
            encoder_state_size, activation="relu")

    def call(self, images_features):
        shape = images_features.get_shape().as_list()
        dim = np.prod(shape[1:])
        input1_images = tf.reshape(images_features, [-1, dim])

        output = self.dense(input1_images)

        # if fine_tuning
        # else
        return output


class Decoder(tf.keras.Model):

    def __init__(self, vocab_size, embedding_size, units):  # the state

        super(Decoder, self).__init__()

        self.embedding = tf.keras.layers.Embedding(
            vocab_size, embedding_size, mask_zero=True)
        self.gru = tf.keras.layers.GRU(units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')
        self.dense = tf.keras.layers.Dense(
            vocab_size, activation="softmax")

    def call(self, x, encoder_features, dec_hidden):

        # x shape == (batch_size,) (giving one word at a time)
        # x shape after passing through embedding == (batch_size, embedding_dim)
        x = self.embedding(x)

        # passing the concatenated vector to the GRU
        # apply GRU output shape == (batch_size, 1, dec_units); dec_hidden shape == (batch_size, dec_units)
        output, dec_hidden = self.gru(tf.expand_dims(x, 1))

        # output shape == (batch_size, dec_units)
        output = tf.reshape(output, (-1, output.shape[2]))

        # output shape == (batch_size, vocab)
        output = self.dense(output)

        return output, dec_hidden

# duvidas o gru nao posmos initial state? e se fosse lstm? ->qual o hidden q davamos? dado ter 2 states?
# tenta fazer tu com o model de acordo cm a explicação
# qual e o initial state da gru?


class SimpleEncoderDecoder(AbstractModel):

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

    def loss_function(self, real, pred):
        print("\n real", real)
        print("predicted", pred)
        print("what is the id of padding token", self.token_to_id[PAD_TOKEN])

        # convert what is not padding [!=0] to True, and padding [==0] to false
        mask = tf.math.logical_not(tf.math.equal(
            np.argmax(real, axis=1), self.token_to_id[PAD_TOKEN]))  # 5,6,7
        #mask = tf.math.logical_not(tf.math.equal(real, 0))

        print(" np argmaz real", np.argmax(real, axis=1))
        print("np shape argmaz real", np.shape(np.argmax(real, axis=1)))

        print("logocial", tf.math.equal(
            np.argmax(real, axis=1), self.token_to_id[PAD_TOKEN]))

        print("mask", mask)
        print("np shape mask", mask)

        # alterar: tens que fazer se o teu one-hot-enconding é zero!!!

        # start, end, palavra, padding
        # False, False, False, True

        # one hot encoding padding: [0,0,0,1]  -> logo tudo é zero, acha q tudo isso é padding...
        # MUDAR!!

        #True, True, True, False
        #print("\nnp shape mask", mask)

        loss_ = self.crossentropy(real, pred)

        # converst True and Falses to 0s and 1s
        mask = tf.cast(mask, dtype=loss_.dtype)
        # loss is multiplied by masks values (1s and 0s), thus filtering the padding (0)
        loss_ *= mask

        print("other loss", loss_)
        print("reduced loss", tf.reduce_mean(loss_))

        # mean/avarage by batch_size
        return tf.reduce_mean(loss_)

    def summary(self):
        pass

    def save(self):
        pass
        # self.model.save(self.get_path())

    # @tf.function
    def train_step(self, img_tensor, input_caption_seq,  target_caption_seq):

        loss = 0
        other_loss = 0

        dec_hidden = tf.zeros((self.args.batch_size, self.units))

        # max_len -1 since the input_seq has max_len -1 (without end token)
        # and target_sentence is max_len -1 (without start token)
        n_tokens = self.max_len-1

        with tf.GradientTape() as tape:

            encoder_features = self.encoder(img_tensor)

            for i in range(n_tokens):

                predicted_output, dec_hidden = self.decoder(
                    input_caption_seq[:, i], encoder_features, dec_hidden)
                loss += self.crossentropy(
                    target_caption_seq[:, i], predicted_output)
                other_loss += self.loss_function(
                    target_caption_seq[:, i], predicted_output)

        batch_loss = (loss / n_tokens)
        batch_other_loss = (other_loss / n_tokens)
        print("this is batch loss", batch_loss)
        print("this is batch_other_loss", batch_other_loss)

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
            tf.print('Epoch {} Loss {:.4f}'.format(epoch,
                                                   epoch_loss))  # N_BATCH -> n_steps
            tf.print('Time taken for 1 epoch {} sec\n'.format(
                time.time() - start))

            # VALIDATION!
            total_loss = 0

            for batch_samples in val_dataset.take(val_steps):
                images_tensor = batch_samples[0]["input_1"]
                input_caption_seq = batch_samples[0]["input_2"]
                target_caption_seq = batch_samples[1]

                loss = 0
                dec_hidden = tf.zeros((self.args.batch_size, self.units))

                # max_len -1 since the input_seq has max_len -1 (without end token)
                # and target_sentence is max_len -1 (without start token)
                n_tokens = self.max_len-1

                encoder_features = self.encoder(images_tensor)

                for i in range(n_tokens):

                    predicted_output, dec_hidden = self.decoder(
                        input_caption_seq[:, i], encoder_features, dec_hidden)
                    loss += self.crossentropy(
                        target_caption_seq[:, i], predicted_output)

                batch_loss = (loss / n_tokens)
                total_loss += batch_loss

            epoch_loss = total_loss/val_steps
            tf.print('Val Loss {:.4f}'.format(
                epoch_loss))  # N_BATCH -> n_steps

            # TODO: test to see if make senses:
            # - run model
            # - refactor (using different methods)
            # TODO: compute callback of save model!
            # TODO: try to do early stop
