import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


def dense_cnns(front_end_output, is_training, num_filt, config, trainable=True):

    # conv layer 1 - adapting dimensions
    front_end_pad = tf.pad(front_end_output, [[0, 0], [3, 3], [0, 0]], "CONSTANT")
    conv1 = tf.layers.conv1d(
        inputs=front_end_pad,
        filters=num_filt,
        kernel_size=7,
        padding="valid",
        activation=tf.nn.relu,
        kernel_initializer=tf.initializers.variance_scaling(scale=2.0, seed=config["seed"]),
        trainable=trainable,
    )

    if not trainable:
        conv1.trainable = False

    bn_conv1 = tf.layers.batch_normalization(conv1, training=is_training, trainable=trainable)

    # conv layer 2 - residual connection
    bn_conv1_pad = tf.pad(bn_conv1, [[0, 0], [3, 3], [0, 0]], "CONSTANT")
    conv2 = tf.layers.conv1d(
        inputs=bn_conv1_pad,
        filters=num_filt,
        kernel_size=7,
        padding="valid",
        activation=tf.nn.relu,
        kernel_initializer=tf.initializers.variance_scaling(scale=2.0, seed=config["seed"]),
        trainable=trainable,
    )
    bn_conv2 = tf.layers.batch_normalization(conv2, training=is_training, trainable=trainable)
    res_conv2 = tf.add(conv2, bn_conv1)

    # conv layer 3 - residual connection
    bn_conv2_pad = tf.pad(res_conv2, [[0, 0], [3, 3], [0, 0]], "CONSTANT")
    conv3 = tf.layers.conv1d(
        inputs=bn_conv2_pad,
        filters=num_filt,
        kernel_size=7,
        padding="valid",
        activation=tf.nn.relu,
        kernel_initializer=tf.initializers.variance_scaling(scale=2.0, seed=config["seed"]),
        trainable=trainable,
    )
    bn_conv3 = tf.layers.batch_normalization(conv3, training=is_training, trainable=trainable)
    res_conv3 = tf.add(conv3, res_conv2)

    return [front_end_output, bn_conv1, res_conv2, res_conv3]

