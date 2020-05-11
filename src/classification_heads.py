import tensorflow as tf
from flip_gradient import flip_gradient
from gradient_projection import gradient_projection


def regular(y, config):
    y = tf.compat.v1.layers.dense(inputs=tf.reshape(y, [-1, 30]),
                                    activation=None,
                                    units=config['num_classes_dataset'],
                                    kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
    return y

def adversarial_type_a(y, config):
    """ type A adversarial heads
    Discriminator conected to the classifier.
    This case is not contemplated for our experiments as is not possible
    to infer the complex discriminator task from the classification output.
    """
    d_ = tf.compat.v1.placeholder(tf.float32, [None, config['discriminator_dimensions']])

    # RevGrad layer
    flipped = flip_gradient(y, config['lambda'])

    # Gradient projection layer
    y, d = gradient_projection(y, flipped)

    y = tf.reshape(y, [-1, config['num_classes_dataset']])
    d = tf.reshape(d, [-1, config['num_classes_dataset']])

    y = tf.compat.v1.layers.dense(inputs=y,
        activation=None,
        units=config['num_classes_dataset'],
        kernel_initializer=tf.contrib.layers.variance_scaling_initializer())

    den = tf.compat.v1.layers.dense(inputs=d,
                units=10,
                activation=tf.nn.relu,
                kernel_initializer=tf.contrib.layers.variance_scaling_initializer())

    d = tf.compat.v1.layers.dense(inputs=den,
        activation=None,
        units=config['discriminator_dimensions'],
        kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
    return y, d, d_

def adversarial_type_b(y, config):
    """ type B adversarial heads
    A common layer conected to the feature extractor with 2 heads,
    one for the classification and other for the discrimination task.
    """
    shared_dense = y
    d_ = tf.compat.v1.placeholder(tf.float32, [None, config['discriminator_dimensions']])

    # RevGrad layer
    flipped = flip_gradient(shared_dense, config['lambda'])

    # Gradient projection layer
    y, d = gradient_projection(shared_dense, flipped)

    y = tf.compat.v1.layers.dense(inputs=tf.reshape(y, [-1, 30]),
                                    activation=None,
                                    units=config['num_classes_dataset'],
                                    kernel_initializer=tf.contrib.layers.variance_scaling_initializer())

    d = tf.compat.v1.layers.dense(inputs=tf.reshape(d, [-1, 30]),
                        activation=None,
                        units=config['discriminator_dimensions'],
                        kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
    return y, d, d_
