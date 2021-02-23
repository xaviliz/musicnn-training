import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from flip_gradient import flip_gradient
from gradient_projection import gradient_projection


def regular(y, config):
    if config["model_number"] == 20:
        # VGGish models already defined their classification head
        return y
    else:
        return tf.layers.dense(
            inputs=y,
            activation=None,
            units=config["num_classes_dataset"],
            kernel_initializer=tf.initializers.variance_scaling(scale=2.0, seed=config["seed"]),
        )
