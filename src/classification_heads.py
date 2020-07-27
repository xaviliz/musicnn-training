import tensorflow as tf
from flip_gradient import flip_gradient
from gradient_projection import gradient_projection


def regular(y, config):
    if config['model_number'] == 20:
        # VGGish models already defined their classificaton head
        return y
    else:
        return tf.compat.v1.layers.dense(inputs=y,
                                    activation=None,
                                    units=config['num_classes_dataset'],
                                    kernel_initializer=tf.contrib.layers.variance_scaling_initializer())

def adversarial_type_a(y, config):
    """ type A adversarial heads
    Discriminator conected to the classifier.
    This case is not contemplated for our experiments as is not possible
    to infer the complex discriminator task from the classification output.
    """
    d_ = tf.compat.v1.placeholder(tf.float32, [None, config['discriminator_dimensions']])

    y = tf.compat.v1.layers.dense(inputs=y,
        activation=None,
        units=config['num_classes_dataset'],
        kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
    
    # RevGrad layer
    lam = tf.placeholder(tf.float32)
    flipped = flip_gradient(y, lam)

    # Gradient projection layer
    y, d = gradient_projection(y, flipped)

    y = tf.reshape(y, [-1, config['num_classes_dataset']])
    d = tf.reshape(d, [-1, config['num_classes_dataset']])

    d = tf.compat.v1.layers.dense(inputs=d,
                units=config['coupling_layer_units'],
                activation=tf.nn.relu,
                kernel_initializer=tf.contrib.layers.variance_scaling_initializer())

    d = tf.compat.v1.layers.dense(inputs=d,
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
    lam = tf.placeholder(tf.float32)
    flipped = flip_gradient(shared_dense, lam)

    # Gradient projection layer
    y, d = gradient_projection(shared_dense, flipped)

    y = tf.compat.v1.layers.dense(inputs=tf.reshape(y, [-1, config['coupling_layer_units']]),
                                    activation=None,
                                    units=config['num_classes_dataset'],
                                    kernel_initializer=tf.contrib.layers.variance_scaling_initializer())

    d = tf.compat.v1.layers.dense(inputs=tf.reshape(d, [-1, config['coupling_layer_units']]),
                        activation=None,
                        units=config['discriminator_dimensions'],
                        kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
    return y, d, d_

def adversarial_grl(y, config):
    """ Adversarial heads with Gradient reversal layer
    A common layer conected to the feature extractor with 2 heads,
    one for the classification and other for the discrimination task.
    """
    coupling_layer = y
    d_ = tf.compat.v1.placeholder(tf.float32, [None, config['discriminator_dimensions']])

    # RevGrad layer
    lam = tf.placeholder(tf.float32)
    flipped = flip_gradient(coupling_layer, lam)

    y = tf.compat.v1.layers.dense(inputs=coupling_layer,
                                  activation=tf.nn.relu,
                                  units=config['coupling_layer_units'],
                                  kernel_initializer=tf.contrib.layers.variance_scaling_initializer())

    y = tf.compat.v1.layers.dense(inputs=y,
                                  activation=None,
                                  units=config['num_classes_dataset'],
                                  kernel_initializer=tf.contrib.layers.variance_scaling_initializer())

    d = tf.compat.v1.layers.dense(inputs=flipped,
                                  activation=tf.nn.relu,
                                  units=config['coupling_layer_units'],
                                  kernel_initializer=tf.contrib.layers.variance_scaling_initializer())

    d = tf.compat.v1.layers.dense(inputs=d,
                                  activation=None,
                                  units=config['discriminator_dimensions'],
                                  kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
    return y, d, d_, lam
