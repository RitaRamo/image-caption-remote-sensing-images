from models.abstract_model import AbstractModel
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding, TimeDistributed
from tensorflow.keras.models import Model
import numpy as np
import tensorflow as tf

from preprocess_data.tokens import START_TOKEN, END_TOKEN


class SimpleFineTunedModel(AbstractModel):

    MODEL_DIRECTORY = "././experiments/results/SimpleFineTunedModel/"

    def __init__(self, args, vocab_size, max_len, encoder_input_size, lstm_units=256, embedding_size=300):
        super().__init__(args, vocab_size, max_len,
                         encoder_input_size, lstm_units, embedding_size)
        self.model = None

    def create(self):
        # Encoder:
        input1_images = Input(shape=self.encoder_input_size, name='input_1')

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

        # a list: [None, 9, 2]
        shape = image.get_shape().as_list()
        dim = np.prod(shape[1:])            # dim = prod(9,2) = 18
        image = tf.reshape(image, [-1, dim])

        images_encoder = Dense(256, activation="relu")(image)

        # Decoder:
        # Input(shape=(None, max_len-1)) #-1 since we cut the train_caption_input[:-1]; and shift to train_target[1:]
        inputs2_captions = Input(shape=(self.max_len-1), name='input_2')
        words_embeddings = Embedding(
            self.vocab_size, self.embedding_size, mask_zero=True)(inputs2_captions)  # vocab_size,
        decoder_hiddens = LSTM(self.lstm_units, return_sequences=True)(
            words_embeddings, initial_state=[images_encoder, images_encoder])
        outputs = TimeDistributed(
            Dense(self.vocab_size, activation='softmax'))(decoder_hiddens)
        # Todo: por time-distributed!!
        self.model = Model(
            inputs=[input1_images, inputs2_captions], outputs=outputs)

    def generate_text(self, input_image, token_to_id, id_to_token):
        if self.model is None:
            self.load()
        # self.model.load_weights("simple_model-weights.h5")

        input_caption = np.zeros((1, self.max_len-1))

        decoder_sentence = ""

        input_caption[:, 0] = token_to_id[START_TOKEN]
        i = 1
        while True:
            print("\ncurretn input", input_caption)

            outputs_tokens = self.model.predict(
                [input_image, input_caption])
            current_output_index = np.argmax(outputs_tokens[0, i-1])
            current_output_token = id_to_token[current_output_index]
            print("token", current_output_token)

            decoder_sentence += " " + current_output_token

            if (current_output_token == END_TOKEN or
                    i >= self.max_len-1):  # chegas ao 35
                break

            input_caption[0, i] = current_output_index

            i += 1

            # input_caption[0,34]= output index
            # enviar esse input, agora tens i=35
            # foi para o outro em 34

            #         [start]
            # [-, , ] [start,-,...]  input[1]=pos_0

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

    # def generate_text(self, input_image, token_to_id, id_to_token):
    #     if self.model is None:
    #         self.load()
    #     # self.model.load_weights("simple_model-weights.h5")

    #     print("this is my model", self.model)
    #     print("this is my model layers", self.model.layers)

    #     input_caption = np.zeros((1, self.max_len-1))

    #     decoder_sentence = ""

    #     input_caption[:, 0] = token_to_id[START_TOKEN]
    #     i = 1
    #     while True:
    #         print("\ncurretn input", input_caption)

    #         outputs_tokens = self.model.predict(
    #             [input_image, input_caption])
    #         current_output_index = np.argmax(outputs_tokens[0, i])
    #         input_caption[0, i] = current_output_index

    #         current_output_token = id_to_token[current_output_index]
    #         print("token", current_output_token)

    #         decoder_sentence += " " + current_output_token

    #         if (current_output_token == END_TOKEN or
    #                 i >= self.max_len-2):
    #             break

    #         i += 1

    #     print("decoded sentence", decoder_sentence)

    #     return decoder_sentence, input_caption
