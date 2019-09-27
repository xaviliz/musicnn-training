import argparse
import os
import json
import pescador
import shared
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import yaml
from argparse import Namespace


TEST_BATCH_SIZE = 64


def evaluation(batch_dispatcher, tf_vars, array_cost, pred_array, id_array, transfer_learning):

    [sess, normalized_y, cost, x, y_, is_train] = tf_vars
    for batch in tqdm(batch_dispatcher):
        if transfer_learning:
            is_training_source, is_training_target = is_train
            pred, cost_pred = sess.run([normalized_y, cost], feed_dict={x: batch['X'], y_: batch['Y'],
                                                                        is_training_source: False, is_training_target: False})
        else:
            pred, cost_pred = sess.run([normalized_y, cost], feed_dict={x: batch['X'], y_: batch['Y'], is_train: False})

        if not array_cost:  # if array_cost is empty, is the first iteration
            pred_array = pred
            id_array = batch['ID']
        else:
            pred_array = np.concatenate((pred_array,pred), axis=0)
            id_array = np.append(id_array,batch['ID'])
        array_cost.append(cost_pred)
    print('predictions', pred_array.shape)
    print('cost', np.mean(array_cost))
    return array_cost, pred_array, id_array


if __name__ == '__main__':

    # which experiment we want to evaluate?
    # Use the -l functionality to ensamble models: python arg.py -l 1234 2345 3456 4567
    parser = argparse.ArgumentParser()
    parser.add_argument('grountruth_file')
    parser.add_argument('index_file')
    parser.add_argument('model_fol')
    parser.add_argument('results_file')
    parser.add_argument('predictions_file')
    parser.add_argument('-t', '--transfer_learning', action='store_true')
    parser.add_argument('-l', '--list', nargs='+', help='List of models to evaluate', required=True)

    args = parser.parse_args()
    models = args.list
    grountruth_file = args.grountruth_file
    index_file = args.index_file
    model_fol = args.model_fol
    results_file = args.results_file
    predictions_file = args.predictions_file
    transfer_learning = args.transfer_learning

    if transfer_learning:
        import train_tl as train
    else:
        import train

    # load all audio representation paths
    [audio_repr_paths, id2audio_repr_path] = shared.load_id2path(index_file)

    # load ground truth
    [ids, id2gt] = shared.load_id2gt(grountruth_file)

    groundtruth_ids = set(ids)
    estimations_ids = set(id2audio_repr_path.keys())

    missing_estimations = groundtruth_ids.difference(estimations_ids)

    if missing_estimations:
        print('{} missing estimations'.format(len(missing_estimations)))
        ids = list(estimations_ids)

    print('# Test set', len(ids))

    array_cost, pred_array, id_array = [], None, None

    for model in models:

        experiment_folder = os.path.join(model_fol, str(model))
        config = json.load(open(os.path.join(experiment_folder, 'config.json')))
        print('Experiment: ' + str(model))
        print('\n' + str(config))

        # pescador: define (finite, batched & parallel) streamer
        pack = [config, 'overlap_sampling', config['n_frames'], False]
        streams = [pescador.Streamer(train.data_gen_abs_path, id, id2audio_repr_path[id], id2gt[id], pack) for id in ids]
        mux_stream = pescador.ChainMux(streams, mode='exhaustive')
        batch_streamer = pescador.Streamer(pescador.buffer_stream, mux_stream, buffer_size=TEST_BATCH_SIZE, partial=True)
        batch_streamer = pescador.ZMQStreamer(batch_streamer)

        # tensorflow: define model and cost
        fuckin_graph = tf.Graph()
        with fuckin_graph.as_default():
            sess = tf.Session()
            if transfer_learning:
                [x, y_, is_training_source, is_training_target, y, normalized_y, cost, cnn3, model_vars] = train.tf_define_model_and_cost(config)
                is_train = [is_training_source, is_training_target]

            else:
                [x, y_, is_train, y, normalized_y, cost] = train.tf_define_model_and_cost(config)

            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            results_folder = experiment_folder + '/'
            saver.restore(sess, results_folder)
            tf_vars = [sess, normalized_y, cost, x, y_, is_train]

            array_cost, pred_array, id_array = evaluation(batch_streamer, tf_vars, array_cost, pred_array, id_array, transfer_learning)
            sess.close()

    print('Predictions computed, now evaluating..')

    y_true, y_pred = shared.average_predictions(pred_array, id_array, ids, id2gt)

    y_true_existing = []
    y_pred_exisiting = []

    for i in range(len(y_true)):
        if np.sum(y_true[i]):
            y_true_existing.append(y_true[i])
            y_pred_exisiting.append(y_pred[i])

    y_true = y_true_existing
    y_pred = y_pred_exisiting

    roc_auc, pr_auc = shared.auc_with_aggergated_predictions(y_true, y_pred)

    acc = shared.compute_accuracy(y_true, y_pred)

    # print experimental results
    print('\nExperiment: ' + str(models))
    print(config)
    print('ROC-AUC: ' + str(roc_auc))
    print('PR-AUC: ' + str(pr_auc))
    print('Acc: ' + str(acc))
    # store experimental results
    to = open(results_file, 'w')
    to.write('Experiment: ' + str(models))
    to.write('\nROC AUC: ' + str(roc_auc))
    to.write('\nPR AUC: ' + str(pr_auc))
    to.write('\nAcc: ' + str(acc))
    to.write('\n')
    to.close()

    predictions = {id: list(pred.astype('float64')) for id, pred in zip(ids, y_pred)}

    with open(predictions_file, 'w') as f:
        json.dump(predictions, f)
