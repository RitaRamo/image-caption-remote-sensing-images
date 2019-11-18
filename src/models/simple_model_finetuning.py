#from models.abstract_model import AbstractModel
from models.simple_model import SimpleModel

from tensorflow.keras.layers import Input, LSTM, Dense, Embedding, TimeDistributed
from tensorflow.keras.models import Model
import numpy as np
import tensorflow as tf

from preprocess_data.tokens import START_TOKEN, END_TOKEN


class SimpleFineTunedModel(SimpleModel):

    MODEL_DIRECTORY = "././experiments/results/SimpleFineTunedModel/"

    def __init__(
        self,
        model_name,
        vocab_size,
        max_len,
        token_to_id,
        id_to_token,
        encoder_input_size,
        embedding_type=None,
        embedding_size=300,
        lstm_units=256
    ):
        super().__init__(model_name, vocab_size, max_len, token_to_id, id_to_token,
                         encoder_input_size, embedding_type, embedding_size, lstm_units)
        self.model = None

    def _get_encoder_state(self, input1_images):
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

        shape = image.get_shape().as_list()
        dim = np.prod(shape[1:])
        image = tf.reshape(image, [-1, dim])

        encoder_state = Dense(256, activation="relu")(image)

        return encoder_state
