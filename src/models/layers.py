import tensorflow as tf


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
