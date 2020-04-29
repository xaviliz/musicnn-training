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
N_FOLDS = config_file.config_train['spec']['n_folds']


def score_predictions(y_true, y_pred, folds, folds_gt, output_file):
    roc_auc, pr_auc = shared.auc_with_aggergated_predictions(y_true, y_pred)
    acc = shared.compute_accuracy(y_true, y_pred)

    accs = []
    for i in range(len(folds)):
        y_true_fold = folds_gt[i]
        y_pred_fold = list(folds[i].values())
        accs.append(shared.compute_accuracy(y_true_fold, y_pred_fold))

    y_true_argmax = [np.argmax(i) for i in y_true]
    y_pred_argmax = [np.argmax(i) for i in y_pred]

    report = classification_report(y_true_argmax, y_pred_argmax)
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
        data[config_file.DATASET] = dict()
        data[config_file.DATASET]['mean'] = acc
        data[config_file.DATASET]['std'] = np.std(accs)

        json.dump(data, fp, indent=4)



if __name__ == '__main__':
    ids = []
    groundtruth = dict()
    estimations = dict()

    folds = []
    for i in range(5):
        groundtruth_file = os.path.join(DATA_FOLDER, 'gt_test_{}.csv'.format(i))
        estimations_file = os.path.join(MODEL_FOLDER, 'predictions_{}.json'.format(i))

        with open(estimations_file) as f:
            fold = json.load(f)
            folds.append(fold)
            estimations.update(fold)

        ids_fold, gt_fold = shared.load_id2gt(groundtruth_file)

        ids += ids_fold
        groundtruth.update(gt_fold)

    # Sanity check
    # num_ids = [int(i.lstrip('ID')) for i in ids]
    # num_ids.sort()
    # assert(num_ids == list(range(num_ids[-1] + 1)))

    groundtruth_ids = set(ids)
    estimations_ids = set(estimations.keys())

    missing_estimations = groundtruth_ids.difference(estimations_ids)

    if missing_estimations:
        print('missing estimations for ids: {}'.format(missing_estimations))
        ids = list(estimations_ids)

    y_true, y_pred = zip(*[(groundtruth[i], estimations[i]) for i in ids])

    fold_gt = []
    fold_est = []
    for i, fold in enumerate(folds):
        keys = ([i for i in fold.keys() if i in groundtruth.keys()])

        fold_est.append({k: v for k, v in fold.items() if k in keys})
        fold_gt.append([groundtruth[k] for k in keys])

    results_file = os.path.join(MODEL_FOLDER, 'results_whole')
    score_predictions(list(y_true),
                      list(y_pred),
                      fold_est,
                      fold_gt,
                      results_file)
