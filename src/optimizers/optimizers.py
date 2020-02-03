from enum import Enum
import tensorflow as tf
from optimizers.adamod import AdaMod


class OptimizerType(Enum):
    ADAM = "adam"
    ADAMOD = "adamod"


def get_optimizer(optimizer_type, learning_rate=0.001):

    if optimizer_type == OptimizerType.ADAMOD.value:
        return AdaMod()
    else:
        return tf.keras.optimizers.Adam(learning_rate)
