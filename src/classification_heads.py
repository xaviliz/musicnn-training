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


def adversarial_type_a(y, config):
    """type A adversarial heads
    Discriminator connected to the classifier.
    This case is not contemplated for our experiments as is not possible
    to infer the complex discriminator task from the classification output.
    """
    d_ = tf.placeholder(tf.float32, [None, config["discriminator_dimensions"]])

    initializer = tf.initializers.variance_scaling(scale=2.0, seed=config["seed"])
    y = tf.layers.dense(
        inputs=y,
        activation=None,
        units=config["num_classes_dataset"],
        kernel_initializer=initializer,
    )

    # RevGrad layer
    lam = tf.placeholder(tf.float32)
    flipped = flip_gradient(y, lam)

    # Gradient projection layer
    y, d = gradient_projection(y, flipped)

    y = tf.reshape(y, [-1, config['num_classes_dataset']])
    d = tf.reshape(d, [-1, config['num_classes_dataset']])

    d = tf.layers.dense(
        inputs=d,
        units=config["coupling_layer_units"],
        activation=tf.nn.relu,
        kernel_initializer=initializer,
    )

    d = tf.layers.dense(
        inputs=d,
        activation=None,
        units=config["discriminator_dimensions"],
        kernel_initializer=initializer,
    )
    return y, d, d_


def adversarial_type_b(y, config):
    """type B adversarial heads
    A common layer connected to the feature extractor with 2 heads,
    one for the classification and other for the discrimination task.
    """
    shared_dense = y
    d_ = tf.placeholder(tf.float32, [None, config["discriminator_dimensions"]])

    # RevGrad layer
    lam = tf.placeholder(tf.float32)
    flipped = flip_gradient(shared_dense, lam)

    # Gradient projection layer
    y, d = gradient_projection(shared_dense, flipped)

    initializer = tf.initializers.variance_scaling(scale=2.0, seed=config["seed"])
    y = tf.layers.dense(
        inputs=tf.reshape(y, [-1, config["coupling_layer_units"]]),
        activation=None,
        units=config["num_classes_dataset"],
        kernel_initializer=initializer,
    )

    d = tf.layers.dense(
        inputs=tf.reshape(d, [-1, config["coupling_layer_units"]]),
        activation=None,
        units=config["discriminator_dimensions"],
        kernel_initializer=initializer,
    )
    return y, d, d_


def adversarial_grl(y, config):
    """Adversarial heads with Gradient reversal layer
    A common layer connected to the feature extractor with 2 heads,
    one for the classification and other for the discrimination task.
    """
    coupling_layer = y
    d_ = tf.placeholder(tf.float32, [None, config["discriminator_dimensions"]])

    # RevGrad layer
    lam = tf.placeholder(tf.float32)
    flipped = flip_gradient(coupling_layer, lam)

    initializer = tf.initializers.variance_scaling(scale=2.0, seed=config["seed"])
    y = tf.layers.dense(
        inputs=coupling_layer,
        activation=tf.nn.relu,
        units=config["coupling_layer_units"],
        kernel_initializer=initializer,
    )

    y = tf.layers.dense(
        inputs=y,
        activation=None,
        units=config["num_classes_dataset"],
        kernel_initializer=initializer,
    )

    d = tf.layers.dense(
        inputs=flipped,
        activation=tf.nn.relu,
        units=config["coupling_layer_units"],
        kernel_initializer=initializer,
    )

    d = tf.layers.dense(
        inputs=d,
        activation=None,
        units=config["discriminator_dimensions"],
        kernel_initializer=initializer,
    )
    return y, d, d_, lam
