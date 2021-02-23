import os
import argparse
import json
import shared
import train
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import yaml
from argparse import Namespace
import json
from sklearn.metrics import classification_report

config_file = Namespace(**yaml.load(open('config_file.yaml')))

MODEL_FOLDER = config_file.MODEL_FOLDER
DATA_FOLDER = config_file.DATA_FOLDER
DATASET = config_file.DATASET
N_FOLDS = config_file.config_train['n_folds']


def score_predictions(results_file, predictions_file):
    ids, folds, predictions, groundtruth = [], [], dict(), dict()

    # get fold-wise predictions
    for i in range(N_FOLDS):
        groundtruth_file = os.path.join(DATA_FOLDER, 'gt_test_{}.csv'.format(i))

        with open('{}_{}.json'.format(predictions_file, i), 'r') as f:
            fold = json.load(f)
            folds.append(fold)
            predictions.update(fold)

        ids_fold, gt_fold = shared.load_id2gt(groundtruth_file)

        ids += ids_fold
        groundtruth.update(gt_fold)

    groundtruth_ids = set(ids)
    predictions_ids = set(predictions.keys())

    # check if there are missing predictions and update ids
    missing_ids = groundtruth_ids.symmetric_difference(predictions_ids)
    if missing_ids:
        print('ids without predictions or groundtruth: {}'.format(missing_ids))
        ids = list(predictions_ids - missing_ids)

    y_true, y_pred = zip(*[(groundtruth[i], predictions[i]) for i in ids])

    fold_gt, fold_pred = [], []
    for i, fold in enumerate(folds):
        keys = ([i for i in fold.keys() if i in groundtruth.keys()])
        fold_pred.append([predictions[k] for k in keys])
        fold_gt.append([groundtruth[k] for k in keys])

    roc_auc, pr_auc, acc, accs, report = get_metrics(y_true, y_pred, fold_gt, fold_pred, folds)

    store_results(results_file, roc_auc, pr_auc, acc, accs, report)

def get_metrics(y_true, y_pred, fold_gt, fold_pred, folds):
    roc_auc, pr_auc = shared.auc_with_aggergated_predictions(y_true, y_pred)
    acc = shared.compute_accuracy(y_true, y_pred)

    accs = []
    for i in range(N_FOLDS):
        y_true_fold = fold_gt[i]
        y_pred_fold = fold_pred[i]
        accs.append(shared.compute_accuracy(y_true_fold, y_pred_fold))

    y_true_argmax = [np.argmax(i) for i in y_true]
    y_pred_argmax = [np.argmax(i) for i in y_pred]

    report = classification_report(y_true_argmax, y_pred_argmax)

    return roc_auc, pr_auc, acc, accs, report

def store_results(output_file, roc_auc, pr_auc, acc, accs, report):
    # print experimental results
    print('ROC-AUC: ' + str(roc_auc))
    print('PR-AUC: ' + str(pr_auc))
    print('Balanced Acc: ' + str(acc))
    print('Balanced Acc STD: ' + str(np.std(accs)))
    print('latext format:')
    print('{:.2f}\\pm{:.2f}'.format(acc, np.std(accs)))
    print('-' * 20)

    # store experimental results
    to = open(output_file, 'w')
    to.write('\nROC AUC: ' + str(roc_auc))
    to.write('\nPR AUC: ' + str(pr_auc))
    to.write('\nAcc: ' + str(acc))
    to.write('\nstd: ' + str(np.std(accs)))
    to.write('\n')
    to.write('Report:\n')
    to.write('{}\n'.format(report))
    to.close()

    output_summary = '/'.join(output_file.split('/')[:-2]) + '/results.json'
    try:
        with open(output_summary, 'r') as fp:
            data = json.load(fp)
    except:
        data = dict()

    with open(output_summary, 'w+') as fp:
        data[DATASET] = dict()
        data[DATASET]['mean'] = acc
        data[DATASET]['std'] = np.std(accs)

        json.dump(data, fp, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--task', help='evaluation type', choices=['regular', 'alterations'], default='regular')
    parser.add_argument('-a', '--alteration', help='alteration', choices=['bpm', 'loudness', 'key'])
    parser.add_argument('-r', '--range', nargs='+', help='range of values to try', type=float)

    args = parser.parse_args()

    task = args.task
    alteration = args.alteration
    alteration_range = args.range

    if task == 'regular':
        results_file = os.path.join(MODEL_FOLDER, 'results_whole')
        predictions_file = os.path.join(MODEL_FOLDER, 'predictions')

        score_predictions(results_file, predictions_file)

    elif task == 'alterations':
        for alteration_value in alteration_range:
            results_file = os.path.join(MODEL_FOLDER,
                'results_{}'.format(alteration),
                '{}_{}'.format(alteration, alteration_value),
                'results_whole')

            predictions_file = os.path.join(MODEL_FOLDER,
                'results_{}'.format(alteration),
                '{}_{}'.format(alteration, alteration_value),
                'predictions')

            score_predictions(results_file, predictions_file)
    else:
        raise Exception('score_predictions: task not implemented')
