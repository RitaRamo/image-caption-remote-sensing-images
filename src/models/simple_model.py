from models.abstract_model import AbstractModel
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding, TimeDistributed
from tensorflow.keras.models import Model
import numpy as np

from preprocess_data.tokens import START_TOKEN, END_TOKEN


class SimpleModel(AbstractModel):

    EPOCHS = 1

    MODEL_DIRECTORY = "././experiments/results/SimpleModel/"

    def __init__(self, args, vocab_size, max_len, encoder_input_size, lstm_units=256, embedding_size=256):
        super().__init__(args, vocab_size, max_len,
                         encoder_input_size, lstm_units, embedding_size)
        self.model = None

    def create(self):
        # Encoder:
        # We could use only the features as encoder, but they go through another Dense layer to have a representation of specific embedding size
        input1_images = Input(shape=self.encoder_input_size, name='input_1')
        # reshape das imagens!!!
        images_encoder = Dense(
            self.lstm_units, activation="relu")(input1_images)

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

    # vais dando a lista com todas as palavras até agora, pelo que não precisas de actualizar o state manualmnte
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

    # def generate_text2(self, input_image, token_to_id, id_to_token):
    #     if self.model is None:
    #         self.load()

    #     print("model layers", self.model.layers)

    #     encoder_inputs = self.model.input[0]   # input_1
    #     encoder_features = self.model.layers[3](encoder_inputs)   # dense
    #     encoder_states = [encoder_features, encoder_features]

    #     encoder_model = Model(encoder_inputs, encoder_states)
    #     print("entrei aqui no encoder")

    #     decoder_inputs = self.model.input[1]   # input_2
    #     # what is latent dim
    #     decoder_state_input_h = Input(shape=(256,), name='input_3')
    #     decoder_state_input_c = Input(shape=(256,), name='input_4')

    #     embeddings_layer = self.model.layers[2]
    #     embeddings = embeddings_layer(decoder_inputs)

    #     decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    #     decoder_lstm = self.model.layers[4]
    #     print("this is decoder lstm", decoder_lstm)

    #     print("embeddin", embeddings)
    #     print("decoder_states_inputs", decoder_states_inputs)
    #     print("separed", decoder_state_input_h)

    #     decoder_outputs, state_h_dec, state_c_dec = decoder_lstm(
    #         embeddings, initial_state=decoder_states_inputs)

    #     decoder_states = [state_h_dec, state_c_dec]

    #     decoder_dense = self.model.layers[5]
    #     decoder_outputs = decoder_dense(decoder_outputs)

    #     decoder_model = Model(
    #         [decoder_inputs] + decoder_states_inputs,
    #         [decoder_outputs] + decoder_states)

    #     # Começar aqui!!
    #     print("vou comecar ")

    #     states_value = encoder_model.predict(input_image)

    #     # Generate empty target sequence of length 1.
    #     input_caption = np.zeros((1, self.max_len))
    #     print("my initial shape", np.shape(input_caption))

    #     input_caption[0, 0] = token_to_id[START_TOKEN]  # start seq

    #     #start + pading

    #     # Sampling loop for a batch of sequences
    #     # (to simplify, here we assume a batch of size 1).
    #     stop_condition = False
    #     decoded_sentence = ''
    #     i = 1
    #     while not stop_condition:
    #         print("entrei no while")

    #         output_tokens, h, c = decoder_model.predict(
    #             [input_caption] + states_value)

    #         # cuidad
    #         outputs_tokens_model = self.model.predict(
    #             [input_image, input_caption])
    #         current_output_index = np.argmax(outputs_tokens_model[0, i])

    #         current_output_token = id_to_token[current_output_index]
    #         print("\n outro model generated", current_output_token)
    #         print("o outro model genera", current_output_index)

    #         # Sample a token
    #         sampled_token_index = np.argmax(output_tokens[0, i])
    #         print("este sampled_token_index", sampled_token_index)

    #         sampled_token = id_to_token[sampled_token_index]
    #         print("este token", sampled_token)

    #         decoded_sentence += " " + sampled_token

    #         # Exit condition: either hit max length
    #         # or find stop character.
    #         if (sampled_token == END_TOKEN or
    #                 i >= self.max_len-1):
    #             stop_condition = True

    #         input_caption[0, i] = sampled_token_index
    #         print("how it torned out", input_caption)

    #         i += 1

    #         # Update states
    #         # states_value = [h, c] -> fica igual ao de baixo, pq tas a enviar a frase toda, se enviasses so a palavra tinahs
    #         # de actualizar manualmente assim com states_values, step a step!

    #     print("\n final decoded", decoded_sentence)

    #     return decoded_sentence, input_caption
