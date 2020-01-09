from models.abstract_model import AbstractModel
from models.layers import _get_embedding_layer
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding, TimeDistributed
from tensorflow.keras.models import Model
import numpy as np

from preprocess_data.tokens import START_TOKEN, END_TOKEN

import spacy
import tensorflow as tf
#from models.layers import lstm


class BasicModel(AbstractModel):

    MODEL_DIRECTORY = "././experiments/results/BasicModel/"

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

    def _checkpoint(self):
        self.checkpoint_path = "./tf_ckpts"
        return tf.train.Checkpoint(loss=tf.Variable(0.0), optimizer=self.model.optimizer, model=self.model)

    def create(self):
        # Encoder:
        # We could use only the features as encoder, but they go through another Dense layer to have a representation of specific embedding size
        input1_images = Input(shape=self.encoder_input_size, name='input_1')
        encoder_state = self._get_encoder_state(input1_images)

        # Decoder:
        # Input(shape=(None, max_len-1)) #-1 since we cut the train_caption_input[:-1]; and shift to train_target[1:]
        inputs2_captions = Input(shape=(self.max_len-1), name='input_2')
        outputs = self._get_decoder_outputs(inputs2_captions, encoder_state)

        self.model = Model(
            inputs=[input1_images, inputs2_captions], outputs=outputs)

    def _get_encoder_state(self, input1_images):
        if not self.args.fine_tuning:  # feature_extraction
            # flatten ex inception: shape=(None, 64, 2048) -> shape=(None, 131072)
            shape = input1_images.get_shape().as_list()
            dim = np.prod(shape[1:])
            input1_images = tf.reshape(input1_images, [-1, dim])

            return Dense(
                self.units, activation="relu")(input1_images)

        else:  # fine_tuning:
            # TODO: MUDA -> AQUI
            base_model = tf.keras.applications.InceptionV3(include_top=False,  # because it is false, its doesnot have the last layer
                                                           weights='imagenet')
            new_input = base_model.input
            hidden_layer = base_model.layers[-1].output

            for layer in base_model.layers[:249]:
                layer.trainable = False
            for layer in base_model.layers[249:]:
                layer.trainable = True

            finetuned_model = tf.keras.Model(new_input, hidden_layer)

            image = finetuned_model(input1_images)

            # flatten ex inception: shape=(None, 299, 299, 3) -> shape=(None, 131072)
            shape = image.get_shape().as_list()
            dim = np.prod(shape[1:])
            image = tf.reshape(image, [-1, dim])

            encoder_state = Dense(256, activation="relu")(image)

            return encoder_state

    def _get_decoder_outputs(self, inputs2_captions, encoder_state):
        words_embeddings = _get_embedding_layer(
            self.embedding_type,  self.vocab_size, self.embedding_size, self.token_to_id)(inputs2_captions)
        decoder_hiddens = LSTM(self.units, return_sequences=True, return_state=False)(
            words_embeddings, initial_state=[encoder_state, encoder_state])
        outputs = TimeDistributed(
            Dense(self.vocab_size, activation='softmax'))(decoder_hiddens)
        return outputs

    # vais dando a lista com todas as palavras até agora, pelo que não precisas de actualizar o state manualmnte

    def generate_text(self, input_image):
        if self.model is None:
            self.load()
        # self.model.load_weights("simple_model-weights.h5")

        input_caption = np.zeros((1, self.max_len-1))

        decoder_sentence = START_TOKEN + " "

        input_caption[:, 0] = self.token_to_id[START_TOKEN]
        i = 1
        while True:  # change to for!
            #print("\ncurretn input", input_caption)

            outputs_tokens = self.model.predict(
                [input_image, input_caption])
            current_output_index = np.argmax(outputs_tokens[0, i-1])
            current_output_token = self.id_to_token[current_output_index]
            #print("token", current_output_token)

            decoder_sentence += " " + current_output_token

            if (current_output_token == END_TOKEN or
                    i >= self.max_len-1):  # chegas ao 35
                break

            input_caption[0, i] = current_output_index

            i += 1

            # input_caption[0,34]= output index
            # enviar esse input, agora tens i=35
            # foi para o outro em 34

            # input:  [start, ,]
            # output: [-, , ] prox_input:[start,-,...]  input[1]=pos_0

            # [-,--,] [start,-,--] input[2]=pos_1
            # (stop dedar ao input) 3; aos 2 paras

            # [-,--,end] [start,-,--, end] -----noo input[3]=pos_2, ms sim no tex=""+ po_2

            # [start,1,2,end_seq]

            # [start,-,--] ->[-,--, end]
            #      3           -> 3

            # [star,_,--, end]

            # print("alternativa",
            #       id_to_token[np.argmax(outputs_tokens[0, i-1])])

            # print("np.shape", np.shape(outputs_tokens))
            # print("e com axis=2", np.argmax(outputs_tokens, axis=2))
            # print("shape com axis=2", np.shape(
            #     np.argmax(outputs_tokens, axis=2)))

            # for a in range(self.max_len-1):
            #     print("oh meu deus é aqui", i, "agora",
            #           np.argmax(outputs_tokens[0, a]))

        print("decoded sentence", decoder_sentence)

        return decoder_sentence  # , input_caption

    def generate_text2(self, input_image):
        if self.model is None:
            self.load()
        # self.model.load_weights("simple_model-weights.h5")

        input_caption = np.zeros((1, self.max_len-1))

        decoder_sentence = START_TOKEN + " "

        input_caption[:, 0] = self.token_to_id[START_TOKEN]
        i = 1
        while True:  # change to for!
            #print("\ncurretn input", input_caption)

            outputs_tokens = self.model.predict(
                [input_image, input_caption])
            current_output_index = np.argmax(outputs_tokens[0, i-1])
            current_output_token = self.id_to_token[current_output_index]
            #print("token", current_output_token)

            decoder_sentence += " " + current_output_token

            if (current_output_token == END_TOKEN or
                    i >= self.max_len-1):  # chegas ao 35
                break

            input_caption[0, i] = current_output_index

            i += 1

        print("decoded sentence", decoder_sentence)

        return decoder_sentence  # , input_caption
