import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding, TimeDistributed
from tensorflow.keras.models import Model, load_model
import numpy as np
END_TOKEN = "<end_seq>"
START_TOKEN = "<start_seq>"

# ver o erro
# chegar ao generator 750... e ver o que se passa!!


def define_model(vocab_size, max_len, features_shape, embedding_size=256):
    # Encoder:
    # We could use only the features as encoder, but they go through another Dense layer to have a representation of specific embedding size
    # TODO: Change 1311072
    # Input(shape=(None, features_shape))
    input1_images = inception(remove_top=0),
    input1_images = flatten(input1_images)

    #input1_images = Input(shape=(131072,), name='input_1')
    images_encoder = Dense(256, activation="relu")(input1_images)

    # Decoder:
    # Input(shape=(None, max_len-1)) #-1 since we cut the train_caption_input[:-1]; and shift to train_target[1:]
    inputs2_captions = Input(shape=(max_len-1), name='input_2')
    words_embeddings = Embedding(
        vocab_size, 300, input_length=max_len-1, mask_zero=True)(inputs2_captions)  # vocab_size,
    decoder_hiddens, _, _ = LSTM(256, return_sequences=True, return_state=True)(
        words_embeddings, initial_state=[images_encoder, images_encoder])
    outputs = TimeDistributed(
        Dense(vocab_size, activation='softmax'))(decoder_hiddens)
    # Todo: por time-distributed!!
    return Model(inputs=[input1_images, inputs2_captions], outputs=outputs)


def generate_text(model, input_image, max_len, token_to_id, id_to_token):

    print("model layers", model.layers)

    encoder_inputs = model.input[0]   # input_1
    encoder_features = model.layers[3](encoder_inputs)   # dense
    encoder_states = [encoder_features, encoder_features]

    encoder_model = Model(encoder_inputs, encoder_states)
    print("entrei aqui no encoder")

    decoder_inputs = model.input[1]   # input_2
    # what is latent dim
    decoder_state_input_h = Input(shape=(256,), name='input_3')
    decoder_state_input_c = Input(shape=(256,), name='input_4')

    embeddings_layer = model.layers[2]
    embeddings = embeddings_layer(decoder_inputs)

    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_lstm = model.layers[4]
    print("this is decoder lstm", decoder_lstm)

    print("embeddin", embeddings)
    print("decoder_states_inputs", decoder_states_inputs)
    print("separed", decoder_state_input_h)

    decoder_outputs, state_h_dec, state_c_dec = decoder_lstm(
        embeddings, initial_state=decoder_states_inputs)

    decoder_states = [state_h_dec, state_c_dec]

    decoder_dense = model.layers[5]
    decoder_outputs = decoder_dense(decoder_outputs)

    decoder_model = Model(
        [decoder_inputs] + decoder_states_inputs,
        [decoder_outputs] + decoder_states)

    # Começar aqui!!
    print("vou comecar ")

    states_value = encoder_model.predict(input_image)

    # Generate empty target sequence of length 1.
    input_caption = np.zeros((1, max_len))
    print("my initial shape", np.shape(input_caption))

    input_caption[0, 0] = token_to_id[START_TOKEN]  # start seq

    #start + pading

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ''
    i = 1
    while not stop_condition:
        print("entrei no while")

        output_tokens, h, c = decoder_model.predict(
            [input_caption] + states_value)

        # cuidad
        outputs_tokens_model = model.predict(
            [input_image, input_caption])
        current_output_index = np.argmax(outputs_tokens_model[0, i])
        current_output_token = id_to_token[current_output_index]
        print("\nthe model generated", current_output_token)
        print("the model generated index", current_output_index)

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, i])
        print("sampled_token_index", sampled_token_index)

        sampled_token = id_to_token[sampled_token_index]
        print("token", sampled_token)

        decoded_sentence += " " + sampled_token

        # Exit condition: either hit max length
        # or find stop character.
        if (sampled_token == END_TOKEN or
                i >= max_len-1):
            stop_condition = True

        input_caption[0, i] = sampled_token_index
        print("how it torned out", input_caption)

        i += 1

        # Update states
        # states_value = [h, c] -> fica igual ao de baixo, pq tas a enviar a frase toda, se enviasses so a palavra tinahs
        # de actualizar manualmente

    print("\n final decoded", decoded_sentence)

    return decoded_sentence, input_caption


def alternative(model, input_image, max_len, token_to_id, id_to_token):

    input_caption = np.zeros((1, max_len))

    decoder_sentence = ""

    input_caption[:, 0] = token_to_id[START_TOKEN]
    i = 1
    while True:
        print("\ncurretn input", input_caption)

        outputs_tokens = model.predict(
            [input_image, input_caption])
        current_output_index = np.argmax(outputs_tokens[0, i])
        input_caption[0, i] = current_output_index

        current_output_token = id_to_token[current_output_index]
        print("token", current_output_token)

        decoder_sentence += " " + current_output_token

        if (current_output_token == END_TOKEN or
                i >= max_len-1):
            break

        i += 1

    print("decoded sentence", decoder_sentence)

    return decoder_sentence, input_caption

    # fazer alternativa hoje

    # def fazer outro modelo -> fazer essa geração!!

    # def generate_test():
    #     load = model.simple(" ")
    #     #start ,000000000
    #     #buscar output i

    #     #mandas start, ouput, #########
    #     #aí tas a pedir para dado o start, da-me o output; (ignoras, ja q ja sabes qual é)
    #     #aí tas a pedir para dado o output, (q teve o start antes), qual o próximo
    # ignoras os proximos
    #      for i in range(1, OUTPUT_LENGTH):
    #         output = model.predict([encoder_input, decoder_input]).argmax(axis=2)
    #         decoder_input[:,i] = output[:,i]

    # if é ==end_sequence stop!
    #     #crear modelo

    # sequence padding!

    # por a alternativa
