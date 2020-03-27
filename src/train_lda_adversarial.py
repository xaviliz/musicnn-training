import argparse
import json
import os
import time
import random
import pescador
import numpy as np
import tensorflow as tf
import models
import config_file, shared
import pickle
import csv
from tensorflow.python.framework import ops
from scipy.fftpack import dct
from tensorflow.keras.backend import binary_crossentropy


def tf_define_model_and_cost(config):
    # tensorflow: define the model
    with tf.name_scope('model'):
        x = tf.compat.v1.placeholder(tf.float32, [None, config['xInput'], config['yInput']])
        y_ = tf.compat.v1.placeholder(tf.float32, [None, config['num_classes_dataset']])
        is_train = tf.compat.v1.placeholder(tf.bool)
        y = models.model_number(x, is_train, config)

        # deal with the reversal gradient case
        if config['model_number'] == 15:
            d_ = tf.compat.v1.placeholder(tf.float32, [None, 2 * config['discriminator_dimensions']])
            y, d = y

        normalized_y = tf.nn.sigmoid(y)
        print(normalized_y.get_shape())
    print('Number of parameters of the model: ' + str(shared.count_params(tf.trainable_variables()))+'\n')

    # tensorflow: define cost function
    with tf.name_scope('metrics'):
        # if you use softmax_cross_entropy be sure that the output of your model has linear units!
        cost = tf.reduce_mean(binary_crossentropy(y_, y))
        if config['weight_decay'] != None:
            vars = tf.trainable_variables()
            lossL2 = tf.add_n([ tf.nn.l2_loss(v) for v in vars if 'kernel' in v.name ])
            cost = cost + config['weight_decay'] * lossL2
            print('L2 norm, weight decay!')

        # add discriminator loss component
        if config['model_number'] == 15:
            t_cost = cost
            d_cost = tf.reduce_mean(binary_crossentropy(d_, d))

            cost = t_cost + d_cost

    # print all trainable variables, for debugging
    model_vars = [v for v in tf.global_variables()]
    for variables in model_vars:
        print(variables)

    if config['model_number'] == 15:
        return [x, y_, is_train, y, normalized_y, (cost, t_cost, d_cost), d_]
    else:
        return [x, y_, is_train, y, normalized_y, cost]


if __name__ == '__main__':

    # load config parameters defined in 'config_file.py'
    parser = argparse.ArgumentParser()
    parser.add_argument('configuration',
                        help='ID in the config_file dictionary')
    parser.add_argument('-s', '--single_batch', action='store_true', help='iterate over a single batch')
    parser.add_argument('-n', '--number_samples', type=int, help='iterate over a just n random samples')
    args = parser.parse_args()
    config = config_file.config_train[args.configuration]
    single_batch = args.single_batch
    number_samples = args.number_samples

    # load config parameters used in 'preprocess_librosa.py',
    config_json = config_file.DATA_FOLDER + config['audio_representation_folder'] + 'config.json'
    with open(config_json, "r") as f:
        params = json.load(f)
    config['audio_rep'] = params

    # set patch parameters
    if config['audio_rep']['type'] == 'waveform':
        raise ValueError('Waveform-based training is not implemented')

    elif config['audio_rep']['spectrogram_type'] == 'mel':
        config['xInput'] = config['n_frames']
        config['yInput'] = config['audio_rep']['n_mels']

    # load audio representation paths
    file_index = config_file.DATA_FOLDER + config['audio_representation_folder'] + 'index.tsv'
    [audio_repr_paths, id2audio_repr_path] = shared.load_id2path(file_index)

    # load training data
    file_ground_truth_train = config_file.DATA_FOLDER + config['gt_train']
    [ids_train, id2gt_train] = shared.load_id2gt(file_ground_truth_train)

    # load validation data
    file_ground_truth_val = config_file.DATA_FOLDER + config['gt_val']
    [ids_val, id2gt_val] = shared.load_id2gt(file_ground_truth_val)

    # load LDA labels
    with open(config_file.DATA_FOLDER + 'msdid2row.tsv', 'r') as f:
        id2lda_row = {row[0]: int(row[1]) for row in csv.reader(f, delimiter='\t')}

    lda_embeddings = np.load(config_file.DATA_FOLDER + 'lda_embeddings.npy')

    # set output
    config['classes_vector'] = list(range(config['num_classes_dataset']))

    print('# Train:', len(ids_train))
    print('# Val:', len(ids_val))
    print('# Classes:', config['classes_vector'])

    # save experimental settings
    experiment_id = str(shared.get_epoch_time()) + args.configuration
    model_folder = config_file.DATA_FOLDER + 'experiments/' + str(experiment_id) + '/'
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
    json.dump(config, open(model_folder + 'config.json', 'w'))
    print('\nConfig file saved: ' + str(config))

    # tensorflow: define model and cost
    if config['model_number'] == 15:
        [x, y_, is_train, y, normalized_y, costs, d_] = tf_define_model_and_cost(config)
        cost, t_cost, d_cost = costs
    else:
        [x, y_, is_train, y, normalized_y, cost] = tf_define_model_and_cost(config)

    # tensorflow: define optimizer
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS) # needed for batchnorm
    with tf.control_dependencies(update_ops):
        lr = tf.placeholder(tf.float32)
        if config['optimizer'] == 'SGD_clip':
            optimizer = tf.train.GradientDescentOptimizer(lr)
            gradients, variables = zip(*optimizer.compute_gradients(cost))
            gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
            train_step = optimizer.apply_gradients(zip(gradients, variables))
        elif config['optimizer'] == 'SGD':
            optimizer = tf.train.GradientDescentOptimizer(lr)
            train_step = optimizer.minimize(cost)
        elif config['optimizer'] == 'Adam':
            optimizer = tf.train.AdamOptimizer(learning_rate=lr)
            train_step = optimizer.minimize(cost)

    sess = tf.InteractiveSession()
    tf.keras.backend.set_session(sess)

    print('\nEXPERIMENT: ', str(experiment_id))
    print('-----------------------------------')

    if single_batch:
        print('Iterating over a single batch')

        if number_samples:
            size = number_samples
        else:
            size = config['batch_size']
        np.random.seed(0)
        ids_train = list(np.array(ids_train)[np.random.randint(0, high=len(ids_train), size=size)])
        ids_val = list(np.array(ids_val)[np.random.randint(0, high=len(ids_val), size=size)])

        config['ids_train'] = ids_train
        config['ids_val'] = ids_val

        # Re-dump config with ids
        json.dump(config, open(model_folder + 'config.json', 'w'))

    # pescador train: define streamer
    train_pack = [config, config['train_sampling'], config['param_train_sampling'], False]
    train_streams = [pescador.Streamer(data_gen, id, id2audio_repr_path[id], lda_embeddings[id2lda_row[id], :], train_pack) for id in ids_train]
    train_mux_stream = pescador.StochasticMux(train_streams, n_active=config['batch_size']*2, rate=None, mode='exhaustive')
    train_batch_streamer = pescador.Streamer(pescador.buffer_stream, train_mux_stream, buffer_size=config['batch_size'], partial=True)
    train_batch_streamer = pescador.ZMQStreamer(train_batch_streamer)

    # pescador val: define streamer
    val_pack = [config, 'overlap_sampling', config['xInput'], False]
    val_streams = [pescador.Streamer(data_gen, id, id2audio_repr_path[id], lda_embeddings[id2lda_row[id], :], val_pack) for id in ids_val]
    val_mux_stream = pescador.ChainMux(val_streams, mode='exhaustive')
    val_batch_streamer = pescador.Streamer(pescador.buffer_stream, val_mux_stream, buffer_size=config['val_batch_size'], partial=True)
    val_batch_streamer = pescador.ZMQStreamer(val_batch_streamer)

    # tensorflow: create a session to run the tensorflow graph
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    if config['load_model'] != None: # restore model weights from previously saved model
        saver.restore(sess, config['load_model']) # end with /!
        print('Pre-trained model loaded!')

    # writing headers of the train_log.tsv
    fy = open(model_folder + 'train_log.tsv', 'a')
    fy.write('Epoch\ttrain_cost\ttrain_t_cost\ttrain_d_cost\tval_cost\tval_t_cost\tval_d_cost\tepoch_time\tlearing_rate\n')
    fy.close()

    # training
    k_patience = 0
    cost_best_model = np.Inf
    tmp_learning_rate = config['learning_rate']
    print('Training started..')
    for i in range(config['epochs']):
        # training: do not train first epoch, to see random weights behaviour
        start_time = time.time()
        array_train_cost = []
        array_train_t_cost = []
        array_train_d_cost = []
        if i != 0:
            for train_batch in train_batch_streamer:
                tf_start = time.time()
                _, train_cost, train_t_cost, train_d_cost = sess.run([train_step, cost, t_cost, d_cost],
                                         feed_dict={x: train_batch['X'], y_: train_batch['Y'], d_: train_batch['D'], lr: tmp_learning_rate, is_train: True})
                array_train_cost.append(train_cost)
                array_train_t_cost.append(train_t_cost)
                array_train_d_cost.append(train_d_cost)

        # validation
        array_val_cost = []
        array_val_t_cost = []
        array_val_d_cost = []
        for val_batch in val_batch_streamer:
            val_cost, val_t_cost, val_d_cost = sess.run([cost, t_cost, d_cost],
                                feed_dict={x: val_batch['X'], y_: val_batch['Y'], d_: val_batch['D'], is_train: False})
            array_val_cost.append(val_cost)
            array_val_t_cost.append(val_t_cost)
            array_val_d_cost.append(val_d_cost)

        # Keep track of average loss of the epoch
        train_cost = np.mean(array_train_cost)
        train_t_cost = np.mean(array_train_t_cost)
        train_d_cost = np.mean(array_train_d_cost)

        val_cost = np.mean(array_val_cost)
        val_t_cost = np.mean(array_val_t_cost)
        val_d_cost = np.mean(array_val_d_cost)
        epoch_time = time.time() - start_time
        fy = open(model_folder + 'train_log.tsv', 'a')
        fy.write('%d\t%g\t%g\t%g\t%g\t%g\t%g\t%gs\t%g\n' % (i+1, train_cost, train_t_cost, train_d_cost, val_cost, val_t_cost, val_d_cost, epoch_time, tmp_learning_rate))
        fy.close()

        # Decrease the learning rate after not improving in the validation set
        if config['patience'] and k_patience >= config['patience']:
            print('Changing learning rate!')
            tmp_learning_rate = tmp_learning_rate / 2
            print(tmp_learning_rate)
            k_patience = 0

        # Early stopping: keep the best model in validation set
        if val_cost >= cost_best_model:
            k_patience += 1
            print('Epoch %d, train cost %g, train task cost %g, train discriminator cost %g, '
                  'val cost %g, val task cost %g, val discriminator cost %g,'
                  'epoch-time %gs, lr %g, time-stamp %s' %
                  (i+1, train_cost, train_t_cost, train_d_cost, val_cost, val_t_cost, val_d_cost, epoch_time, tmp_learning_rate,
                   str(time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime()))))

        else:
            # save model weights to disk
            save_path = saver.save(sess, model_folder)
            print('Epoch %d, train cost %g, train task cost %g, train discriminator cost %g, '
                  'val cost %g, val task cost %g, val discriminator cost %g,'
                  'epoch-time %gs, lr %g, time-stamp %s - [BEST MODEL]'
                  ' saved in: %s' %
                  (i+1, train_cost, train_t_cost, train_d_cost, val_cost, val_t_cost, val_d_cost, epoch_time,tmp_learning_rate,
                   str(time.strftime('%Y-%m-%d %H:%M:%S',time.gmtime())), save_path))
            cost_best_model = val_cost

    print('\nEVALUATE EXPERIMENT -> '+ str(experiment_id))
