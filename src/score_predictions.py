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

config_file = Namespace(**yaml.load(open('config_file.yaml')))

MODEL_FOLDER = config_file.MODEL_FOLDER
DATA_FOLDER = config_file.DATA_FOLDER
N_FOLDS = config_file.config_train['spec']['n_folds']


def score_predictions(y_true, y_pred, output_file):
    roc_auc, pr_auc = shared.auc_with_aggergated_predictions(y_true, y_pred)
    acc = shared.compute_accuracy(y_true, y_pred)

    # print experimental results
    print('ROC-AUC: ' + str(roc_auc))
    print('PR-AUC: ' + str(pr_auc))
    print('Acc: ' + str(acc))
    # store experimental results
    to = open(output_file, 'w')
    to.write('\nROC AUC: ' + str(roc_auc))
    to.write('\nPR AUC: ' + str(pr_auc))
    to.write('\nAcc: ' + str(acc))
    to.write('\n')
    to.close()

if __name__ == '__main__':
    ids = []
    groundtruth = dict()
    estimations = dict()

    for i in range(N_FOLDS):
        groundtruth_file = os.path.join(DATA_FOLDER, 'gt_test_{}.csv'.format(i))
        estimations_file = os.path.join(MODEL_FOLDER, 'predictions_{}.json'.format(i))

        with open(estimations_file) as f:
            estimations.update(json.load(f))

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

    results_file = os.path.join(MODEL_FOLDER, 'results_whole')
    score_predictions(list(y_true),
                      list(y_pred),
                      results_file)
