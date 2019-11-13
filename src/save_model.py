import argparse
import os
import json
import pescador
import shared
import train
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import yaml
from argparse import Namespace


if __name__ == '__main__':
    # which experiment we want to evaluate?
    # Use the -l functionality to ensamble models: python arg.py -l 1234 2345 3456 4567
    parser = argparse.ArgumentParser()
    parser.add_argument('model_fol')
    parser.add_argument('-l', '--list', nargs='+', help='List of models to evaluate', required=True)

    args = parser.parse_args()
    models = args.list
    model_fol = args.model_fol

    for model in models:
        experiment_folder = os.path.join(model_fol, str(model))
        config = json.load(open(os.path.join(experiment_folder, 'config.json')))
        print('Experiment: ' + str(model))
        print('\n' + str(config))

        # pescador: define (finite, batched & parallel) streamer

        # tensorflow: define model and cost
        # fuckin_graph = tf.Graph()

        output_graph = os.path.join(model_fol, 'frozen.pb')

        # with fuckin_graph.as_default():
        sess = tf.Session()

        [x, y_, is_train, y, normalized_y, cost, model_vars] = train.tf_define_model_and_cost(config)
        sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver()
        results_folder = experiment_folder + '/'
        saver.restore(sess, results_folder)

        gd = sess.graph.as_graph_def()
        # for node in gd.node:
        #     if node.op == 'RefSwitch':
        #         node.op = 'Switch'
        #         for index in range(len(node.input)):
        #             if 'moving_' in node.input[index]:
        #                 node.input[index] = node.input[index] + '/read'
        #     elif node.op == 'AssignSub':
        #         node.op = 'Sub'
        #         if 'use_locking' in node.attr: del node.attr['use_locking']
        #     elif node.op == 'AssignAdd':
        #         node.op = 'Add'
        #        if 'use_locking' in node.attr: del node.attr['use_locking']

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
            
        node_names =[n.name for n in gd.node if 'model' in n.name]
        print(node_names)
        subgraph = tf.graph_util.extract_sub_graph(gd, node_names)
        tf.reset_default_graph()
        tf.import_graph_def(subgraph)
        # node_names =['dense_1']


        output_graph_def = tf.graph_util.convert_variables_to_constants(
                sess, # The session is used to retrieve the weights
                gd, # The graph_def is used to retrieve the nodes 
                node_names # The output node names are used to select the usefull nodes
            )
        tf.io.write_graph(output_graph_def, model_fol, output_graph, as_text=False)
        sess.close()

            # Finally we serialize and dump the output graph to the filesystem

            # with tf.gfile.GFile(output_graph, "wb") as f:
            #     f.write(output_graph_def.SerializeToString())
            # print("%d ops in the final graph." % len(output_graph_def.node))

