import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


def musically_motivated_cnns(x, is_training, yInput, num_filt, config, type, trainable=True):

    expanded_layer = tf.expand_dims(x, 3)
    input_layer = tf.layers.batch_normalization(expanded_layer, training=is_training, trainable=trainable)

    input_pad_7 = tf.pad(input_layer, [[0, 0], [3, 3], [0, 0], [0, 0]], "CONSTANT")

    if 'timbral' in type:

        # padding only time domain for an efficient 'same' implementation
        # (since we pool throughout all frequency afterwards)
        input_pad_7 = tf.pad(input_layer, [[0, 0], [3, 3], [0, 0], [0, 0]], "CONSTANT")

        if '74' in type:
            f74 = timbral_block(inputs=input_pad_7,
                           filters=int(num_filt*128),
                           kernel_size=[7, int(0.4 * yInput)],
                           is_training=is_training,
                           trainable=trainable,
                           config=config)

        if '77' in type:
            f77 = timbral_block(inputs=input_pad_7,
                           filters=int(num_filt*128),
                           kernel_size=[7, int(0.7 * yInput)],
                           is_training=is_training,
                           trainable=trainable,
                           config=config)

    if 'temporal' in type:

        s1 = tempo_block(inputs=input_layer,
                          filters=int(num_filt*32),
                          kernel_size=[128,1],
                          is_training=is_training,
                          trainable=trainable,
                          config=config)

        s2 = tempo_block(inputs=input_layer,
                          filters=int(num_filt*32),
                          kernel_size=[64,1],
                          is_training=is_training,
                          trainable=trainable,
                          config=config)

        s3 = tempo_block(inputs=input_layer,
                          filters=int(num_filt*32),
                          kernel_size=[32,1],
                          is_training=is_training,
                          trainable=trainable,
                          config=config)

    # choose the feature maps we want to use for the experiment
    if type == '7774timbraltemporal':
        return [f74, f77, s1, s2, s3]

    elif type == '74timbral':
        return [f74]


def timbral_block(inputs, filters, kernel_size, is_training, config, padding="valid", activation=tf.nn.relu, trainable=True):

    conv = tf.layers.conv2d(
        inputs=inputs,
        filters=filters,
        kernel_size=kernel_size,
        padding=padding,
        activation=activation,
        trainable=trainable,
        kernel_initializer=tf.initializers.variance_scaling(scale=2.0, seed=config["seed"]),
    )
    bn_conv = tf.layers.batch_normalization(conv, training=is_training, trainable=trainable)
    pool = tf.layers.max_pooling2d(inputs=bn_conv, pool_size=[1, bn_conv.shape[2]], strides=[1, bn_conv.shape[2]])
    return tf.squeeze(pool, [2])


def tempo_block(inputs, filters, kernel_size, is_training, config, padding="same", activation=tf.nn.relu, trainable=True):

    conv = tf.layers.conv2d(
        inputs=inputs,
        filters=filters,
        kernel_size=kernel_size,
        padding=padding,
        activation=activation,
        trainable=trainable,
        kernel_initializer=tf.initializers.variance_scaling(scale=2.0, seed=config["seed"]),
    )
    bn_conv = tf.layers.batch_normalization(conv, training=is_training, trainable=trainable)
    pool = tf.layers.max_pooling2d(inputs=bn_conv, pool_size=[1, bn_conv.shape[2]], strides=[1, bn_conv.shape[2]])
    return tf.squeeze(pool, [2])

