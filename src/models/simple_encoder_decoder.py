import tensorflow as tf
from models.abstract_model import AbstractModel

# TODO: por mask!!
# TODO: por save() -> como defines isso??


class Encoder(tf.keras.Model):
    def __init__(self, encoder_state_size):
        super(Encoder, self).__init__()
        # inception
        self.dense = tf.keras.layers.Dense(
            encoder_state_size, activation="relu")

    def call(self, images_features):
        # inception
        # reshape
        output = self.dense(images_features)
        return output


class Decoder(tf.keras.Model):

    def __init__(self, vocab_size, embedding_size, state_size):  # the state

        super(Decoder, self).__init__()
        self.embedding = tf.keras.layers.Embedding(
            vocab_size, embedding_size)
        # LSTM units should have the same size as the initial_state (which should be equal to size of the encoder state)
        self.lstm = tf.keras.layers.LSTM(
            state_size, return_sequences=True, return_state=True)
        self.dense = tf.keras.layers.Dense(
            vocab_size, activation="softmax")

    def call(self, sequence, encoder_state):

        embed = self.embedding(sequence)
        lstm_out, state_h, state_c = self.lstm(
            embed, initial_state=encoder_state)
        outputs = self.dense(lstm_out)
        return outputs, state_h, state_c


class SimpleEncoderDecoderModel(AbstractModel):

    def __init__(self, vocab_size, max_len, encoder_input_size=131072, lstm_units=256, embedding_size=300):
        super().__init__(vocab_size, max_len, encoder_input_size, lstm_units, embedding_size)
        self.optimizer = tf.keras.optimizers.Adam()
        self.encoder = Encoder(lstm_units)
        self.decoder = Decoder(vocab_size, embedding_size, lstm_units)

    def create(self):
        pass

    def build(self):
        pass

    def loss_func(self, targets, predicted):

        crossentropy = tf.keras.losses.CategoricalCrossentropy()  # from_logits=True)
        # mask = tf.math.logical_not(tf.math.equal(targets, 0))
        # mask = tf.cast(mask, dtype=tf.int64)
        # loss = crossentropy(targets, logits, sample_weight=mask)
        loss = crossentropy(targets, predicted)

        return loss

    def train_step(self,  images, input_caption_seq, target_caption_seq):

        with tf.GradientTape() as tape:
            encoder_state = self.encoder(images)

            decoder_predicted, _, _ = self.decoder(
                input_caption_seq, (encoder_state, encoder_state))

            loss = self.loss_func(target_caption_seq, decoder_predicted)

        variables = self.encoder.trainable_variables + self.decoder.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))

        return loss

    def train(self, train_dataset, val_dataset, len_train_dataset, len_val_dataset):

        NUM_EPOCHS = 2
        for e in range(NUM_EPOCHS):
            print("entrei")
            for batch, batch_samples in enumerate(train_dataset):
                images = batch_samples[0]["input_1"]
                input_caption_seq = batch_samples[0]["input_2"]
                target_caption_seq = batch_samples[1]

                loss = self.train_step(images, input_caption_seq,
                                       target_caption_seq)
                if batch >= 5:
                    break
                print("batch loss", loss)
        print("acabou", batch)

    # def generate!!!
    # ver se funca!!
