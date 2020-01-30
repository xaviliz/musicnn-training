import os
import yaml
import argparse
from argparse import Namespace

import tensorflow as tf
import shared

config_file = Namespace(**yaml.load(open('config_file.yaml'), Loader=yaml.SafeLoader))


def tf_define_model(config):
    with tf.name_scope('model'):
        x = tf.compat.v1.placeholder(tf.float32, [None, config['xInput'], config['yInput']])

        # If load_model is defined, use pre-trained models
        if config['load_model'] is not None:
            import models_transfer_learning as models
            y = models.define_model(x, False, config['model_number'],
                                    config['num_classes_dataset'])
        else:
            import models
            y = models.model_number(x, False, config)

        normalized_y = tf.nn.sigmoid(y)
        print(normalized_y.get_shape())
    print('Number of parameters of the model: ' + str(shared.count_params(tf.trainable_variables())) + '\n')

    # Print all trainable variables, for debugging
    model_vars = [v for v in tf.global_variables()]
    for variables in model_vars:
        print(variables)

    return [x, y, normalized_y, model_vars]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--output_graph_name', '-o',
                        help='output protobuff (.pb) file name',
                        default='small_vggish_init.pb')

    args = parser.parse_args()
    output_graph = args.output_graph_name

    config = config_file.config_train['spec']
    config['audio_rep'] = {'type': 'spec',
                           'spectrogram_type': 'mel',
                           'n_mels': 64
                           }

    config['xInput'] = config['n_frames']
    config['yInput'] = config['audio_rep']['n_mels']

    model_folder = config_file.MODEL_FOLDER

    # Define model
    tf_define_model(config)

    sess = tf.InteractiveSession()
    tf.keras.backend.set_session(sess)

    print('\nInitialazing graph')
    print('-----------------------------------')
    sess.run(tf.global_variables_initializer())
    gd = sess.graph.as_graph_def()

    # Ugly fix. Needed for serialization
    for node in gd.node:
        if node.op == 'RefSwitch':
            node.op = 'Switch'
            for index in range(len(node.input)):
                if 'moving_' in node.input[index]:
                    node.input[index] = node.input[index] + '/read'
        elif node.op == 'AssignSub':
            node.op = 'Sub'
            if 'use_locking' in node.attr: del node.attr['use_locking']
        elif node.op == 'AssignAdd':
            node.op = 'Add'
            if 'use_locking' in node.attr: del node.attr['use_locking']
        elif node.op == 'Assign':
            node.op = 'Identity'
            if 'use_locking' in node.attr: del node.attr['use_locking']
            if 'validate_shape' in node.attr: del node.attr['validate_shape']
            if len(node.input) == 2:
                # input0: ref: Should be from a Variable node. May be uninitialized.
                # input1: value: The value to be assigned to the variable.
                node.input[0] = node.input[1]
                del node.input[1]

    # Extract only the nodes in the model scope
    node_names = [n.name for n in gd.node if 'model' in n.name]

    subgraph = tf.graph_util.extract_sub_graph(gd, node_names)
    tf.reset_default_graph()
    tf.import_graph_def(subgraph)

    print('\nSerializing graph')
    print('-----------------------------------')
    output_graph_def = tf.graph_util.convert_variables_to_constants(
        sess,  # The session is used to retrieve the weights
        gd,  # The graph_def is used to retrieve the nodes
        node_names  # The output node names are used to select the usefull nodes
    )

    print('\nSaving graph')
    print('-----------------------------------')
    tf.io.write_graph(output_graph_def, model_folder, output_graph, as_text=False)
    sess.close()
