import argparse
import json
import pescador
import shared, train
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import yaml
from argparse import Namespace
from data_loaders import data_gen_standard as data_gen
import os

config_file = Namespace(**yaml.load(open('config_file.yaml'), Loader=yaml.SafeLoader))

TEST_BATCH_SIZE = 64
MODEL_FOLDER = config_file.MODEL_FOLDER
DATA_FOLDER = config_file.DATA_FOLDER
FILE_INDEX = DATA_FOLDER + 'index_repr.tsv'
FILE_GROUND_TRUTH_TEST = config_file.config_train['gt_test']
FOLD = config_file.config_train['fold']
NUM_CLASSES_DATASET = config_file.config_train['num_classes_dataset']


def get_lowlevel_descriptors_groundtruth(config, id2audio_repr_path, orig_ids):
    print('Changing groundtruth to "{}" values'.format(config['lowlevel_descriptor']))
    global NUM_CLASSES_DATASET

    if 'loudness' in config['lowlevel_descriptor']:
        from feature_functions import get_loudness as gt_extractor
        NUM_CLASSES_DATASET = 1
    elif 'bpm' in config['lowlevel_descriptor']:
        from feature_functions import get_bpm as gt_extractor
        NUM_CLASSES_DATASET = 1
    elif 'scale' in config['lowlevel_descriptor']:
        from feature_functions import get_mode as gt_extractor
        NUM_CLASSES_DATASET = 1
    elif 'key' in config['lowlevel_descriptor']:
        from feature_functions import get_key as gt_extractor
        NUM_CLASSES_DATASET = 12

    data_folder = os.path.dirname(config['gt_test'])

    ids = []
    id2gt = dict()

    for oid in orig_ids:
        try:
            id2gt[oid] = gt_extractor(data_folder,
                                      os.path.join(data_folder, id2audio_repr_path[oid]))
            ids.append(oid)
        except:
            print('{} failed'.format(oid))

    return ids, id2gt

def prediction(config, experiment_folder, id2audio_repr_path, id2gt, ids):
    # pescador: define (finite, batched & parallel) streamer
    pack = [config, 'overlap_sampling', config['n_frames'], False, DATA_FOLDER]
    streams = [pescador.Streamer(data_gen, id, id2audio_repr_path[id], id2gt[id], pack) for id in ids]
    mux_stream = pescador.ChainMux(streams, mode='exhaustive')
    batch_streamer = pescador.Streamer(pescador.buffer_stream, mux_stream, buffer_size=TEST_BATCH_SIZE, partial=True)
    batch_streamer = pescador.ZMQStreamer(batch_streamer)

    # tensorflow: define model and cost
    fuckin_graph = tf.Graph()
    with fuckin_graph.as_default():
        sess = tf.Session()
        [x, y_, is_train, y, normalized_y, cost, _] = train.tf_define_model_and_cost(config)
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(sess, experiment_folder)

        pred_array, id_array = np.empty([0, NUM_CLASSES_DATASET]), np.empty(0)
        for batch in tqdm(batch_streamer):
            pred, _ = sess.run([normalized_y, cost], feed_dict={x: batch['X'], y_: batch['Y'], is_train: False})
            # make shure our predictions have are a np
            # array with the proper shape
            pred = np.array(pred).reshape(-1, NUM_CLASSES_DATASET)
            pred_array = np.vstack([pred_array, pred])
            id_array = np.hstack([id_array, batch['ID']])

        sess.close()

    print(pred_array.shape)
    print(id_array.shape)

    print('Predictions computed, now evaluating...')
    y_true, y_pred = shared.average_predictions(pred_array, id_array, ids, id2gt)
    roc_auc, pr_auc = shared.auc_with_aggergated_predictions(y_true, y_pred)
    acc = shared.compute_accuracy(y_true, y_pred)

    metrics = (roc_auc, pr_auc, acc)
    return y_pred, metrics

def store_results(config, results_file, predictions_file, models, ids, y_pred, metrics):
    roc_auc, pr_auc, acc = metrics

    results_folder = os.path.dirname(results_file)
    os.makedirs(results_folder, exist_ok=True)

    # print experimental results
    print('Metrics:')
    print('ROC-AUC: ' + str(roc_auc))
    print('PR-AUC: ' + str(pr_auc))
    print('Acc: ' + str(acc))

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


if __name__ == '__main__':
    # which experiment we want to evaluate?
    # Use the -l functionality to ensamble models: python arg.py -l 1234 2345 3456 4567
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--list', nargs='+', help='List of models to evaluate', required=True)
    parser.add_argument('-t', '--task', help='evaluation type', choices=['regular', 'alterations'], default='regular')
    parser.add_argument('-a', '--alteration', help='alteration', choices=['bpm', 'loudness', 'key'])
    parser.add_argument('-r', '--range', nargs='+', help='range of values to try', type=float)

    args = parser.parse_args()

    models = args.list
    task = args.task
    alteration = args.alteration
    alteration_range = args.range

    # load all audio representation paths
    [audio_repr_paths, id2audio_repr_path] = shared.load_id2path(FILE_INDEX)

    for model in models:
        experiment_folder = MODEL_FOLDER + 'experiments/' + str(model) + '/'
        config = json.load(open(experiment_folder + 'config.json'))

        print('Experiment: ' + str(model))
        print('\n' + str(config))

        # Don't load auxiliary (adversarial) modules for evaluation
        config['mode'] = 'regular'

        # modify the configuration if required to inform the dataloader
        if task == 'alterations':
            config['task'] = 'alterations'
            config['alteration'] = alteration

        # load ground truth
        print('groundtruth file: {}'.format(FILE_GROUND_TRUTH_TEST))
        ids, id2gt = shared.load_id2gt(FILE_GROUND_TRUTH_TEST)

        if config['task'] == 'lowlevel_descriptors':
            ids, id2gt = get_lowlevel_descriptors_groundtruth(config, id2audio_repr_path, ids)


        print('# Test set size: ', len(ids))

        if task == 'regular':
            print('Performing regular evaluation')
            y_pred, metrics = prediction(config, experiment_folder, id2audio_repr_path, id2gt, ids)

            # store experimental results
            results_file = os.path.join(MODEL_FOLDER, 'results_{}'.format(FOLD))
            predictions_file = os.path.join(MODEL_FOLDER, 'predictions_{}.json'.format(FOLD))

            store_results(config, results_file, predictions_file, models, ids, y_pred, metrics)

        elif task == 'alterations':
            print('Performing alterations evaluation')

            for alteration_value in alteration_range:
                print('Alterating {} to a value of {}'.format(alteration, alteration_value))
                config['alteration_value'] = alteration_value

                y_pred, metrics = prediction(config, experiment_folder, id2audio_repr_path, id2gt, ids)

                # store experimental results
                results_file = os.path.join(MODEL_FOLDER,
                                            'results_{}'.format(alteration),
                                            '{}_{}'.format(alteration, alteration_value),
                                            'results_{}'.format(FOLD))

                predictions_file = os.path.join(MODEL_FOLDER,
                                                'results_{}'.format(alteration),
                                                '{}_{}'.format(alteration, alteration_value),
                                                'predictions_{}.json'.format(FOLD))

                store_results(config, results_file, predictions_file, models, ids, y_pred, metrics)
        else:
            raise Exception('Evaluation type "{}" is not implemented'.format(task))
