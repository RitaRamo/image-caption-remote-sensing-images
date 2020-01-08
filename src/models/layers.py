import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Embedding
import spacy


def _get_embedding_layer(embedding_type, vocab_size, embedding_size, token_to_id):
    if embedding_type is None:
        return Embedding(vocab_size, embedding_size, mask_zero=True)

    embeddings_matrix = None

    if embedding_type == "glove":

        glove_path = 'src/models/glove.6B/glove.6B.300d.txt'
        embedding_size = 300

        def read_glove_vectors(path, lenght):
            embeddings = {}
            with open(path) as glove_f:
                for line in glove_f:
                    chunks = line.split()
                    word = chunks[0]
                    vector = np.array(chunks[1:])
                    embeddings[word] = vector

            return embeddings

        glove_embeddings = read_glove_vectors(
            glove_path, embedding_size)

        # Init the embeddings layer with GloVe embeddings
        embeddings_matrix = np.zeros(
            (vocab_size, embedding_size))
        for word, idx in token_to_id.items():
            try:
                embeddings_matrix[idx] = glove_embeddings[word]
            except:
                pass

    elif embedding_type == "spacy":
        nlp = spacy.load('en_core_web_md')
        embedding_size = len(nlp.vocab['apple'].vector)

        embeddings_matrix = np.zeros(
            (vocab_size, embedding_size))
        for word, i in token_to_id.items():
            try:
                embeddings_matrix[i] = nlp.vocab[word].vector
            except:
                pass

    return Embedding(vocab_size,
                     embedding_size,
                     mask_zero=True,
                     weights=[embeddings_matrix],
                     trainable=False)


def gru(units, return_sequences=True, return_state=True, initializer='glorot_uniform'):
  # If you have a GPU, we recommend using CuDNNGRU(provides a 3x speedup than GRU)
  # the code automatically does that.
    if len(
        tf.config.experimental.list_physical_devices('GPU')
    ) > 0:
        return tf.keras.layers.CuDNNGRU(units,
                                        return_sequences=return_sequences,
                                        return_state=return_state,
                                        recurrent_initializer=initializer)
    else:
        return tf.keras.layers.GRU(units,
                                   return_sequences=return_sequences,
                                   return_state=return_state,
                                   recurrent_initializer=initializer)


def lstm(units, return_sequences=True, return_state=False, initializer='orthogonal'):

    if len(
        tf.config.experimental.list_physical_devices('GPU')
    ) > 0:
        return tf.keras.layers.CuDNNLSTM(units,
                                         return_sequences=return_sequences,
                                         return_state=return_state,
                                         recurrent_initializer=initializer)
    else:
        return tf.keras.layers.LSTM(units,
                                    return_sequences=return_sequences,
                                    return_state=return_state,
                                    recurrent_initializer=initializer)

# por isto e tudo
# chamar
# ver se o score foi o memso dos resultados!
