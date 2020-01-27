from enum import Enum
import tensorflow as tf
from optimizers.adamod import AdaMod


class OptimizerType(Enum):
    ADAM = "adam"
    ADAMOD = "adamod"


def get_optimizer(optimizer_type):

    if optimizer_type == OptimizerType.ADAMOD.value:
        return AdaMod()
    else:
        return tf.keras.optimizers.Adam()
