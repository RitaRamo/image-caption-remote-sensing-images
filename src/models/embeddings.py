import numpy as np
from enum import Enum
import spacy
from preprocess_data.tokens import END_TOKEN


class EmbeddingsType(Enum):
    GLOVE = "glove"
    SPACY = "spacy"


def read_glove_vectors(path, lenght):
    embeddings = {}
    with open(path) as glove_f:
        for line in glove_f:
            chunks = line.split()
            word = chunks[0]
            vector = np.array(chunks[1:])
            embeddings[word] = vector

    return embeddings


def get_glove_embeddings_matrix(vocab_size, embedding_size, token_to_id):
    # ter caderno, fazer a logica, ver outros problemas...

    glove_path = 'src/models/glove.6B/glove.6B.'+str(embedding_size) + 'd.txt'

    glove_embeddings = read_glove_vectors(
        glove_path, embedding_size)

    # Init the embeddings layer
    embeddings_matrix = np.zeros(
        (vocab_size, embedding_size))
    for word, id in token_to_id.items():
        try:
            embeddings_matrix[id] = glove_embeddings[word]
        except:
            pass

    return embeddings_matrix


def get_glove_embeddings_matrix_for_continuous(vocab_size, embedding_size, token_to_id):
    # ter caderno, fazer a logica, ver outros problemas...

    glove_path = 'src/models/glove.6B/glove.6B.'+str(embedding_size) + 'd.txt'

    glove_embeddings = read_glove_vectors(
        glove_path, embedding_size)

    embedding_size = embedding_size+1  # add one dim for the END_TOKEN

    # Init the embeddings layer
    embeddings_matrix = np.zeros(
        (vocab_size, embedding_size))

    embeddings_matrix[token_to_id[END_TOKEN], -1] = 1

    for word, id in token_to_id.items():
        try:
            embeddings_matrix[id, :-1] = glove_embeddings[word]
        except:
            pass

    return embeddings_matrix, embedding_size


def read_spacy_embeddings_and_dim(path, lenght):
    nlp = spacy.load('en_core_web_md')
    embedding_size = len(nlp.vocab['apple'].vector)

    return nlp, embedding_size


def get_spacy_embeddings_matrix_and_dim(vocab_size, token_to_id):

    nlp = spacy.load('en_core_web_md')
    embedding_size = len(nlp.vocab['apple'].vector)

    embeddings_matrix = np.zeros(
        (vocab_size, embedding_size))
    for word, id in token_to_id.items():
        try:
            embeddings_matrix[id] = nlp.vocab[word].vector
        except:
            pass

    return embeddings_matrix, embedding_size
