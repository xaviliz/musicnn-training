import json
import os
import argparse
from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report

import shared


def score_predictions(args):
    config = json.load(open(Path(args.config_file), 'r'))

    results_file = Path(config['exp_dir'], 'results_whole')
    predictions_file = Path(config['exp_dir'], 'predictions')

    ids, folds, predictions, groundtruth = [], [], dict(), dict()

    n_folds = config['config_train']['n_folds']
    data_dir = Path(config['data_dir'])
    dataset = config['dataset']

    # get fold-wise predictions
    for i in range(n_folds):
        groundtruth_file = data_dir / f"gt_test_{i}.csv"

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

    roc_auc, pr_auc, acc, accs, report = get_metrics(
        y_true, y_pred, fold_gt, fold_pred, n_folds)

    store_results(results_file, roc_auc, pr_auc, acc, accs, report, dataset)


def get_metrics(y_true, y_pred, fold_gt, fold_pred, n_folds):
    roc_auc, pr_auc = shared.auc_with_aggergated_predictions(y_true, y_pred)
    acc = shared.compute_accuracy(y_true, y_pred)

    accs = []
    for i in range(n_folds):
        y_true_fold = fold_gt[i]
        y_pred_fold = fold_pred[i]
        accs.append(shared.compute_accuracy(y_true_fold, y_pred_fold))

    y_true_argmax = [np.argmax(i) for i in y_true]
    y_pred_argmax = [np.argmax(i) for i in y_pred]

    report = classification_report(y_true_argmax, y_pred_argmax)

    return roc_auc, pr_auc, acc, accs, report


def store_results(output_file, roc_auc, pr_auc, acc, accs, report, dataset):
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

    output_summary = output_file.parent.parent / 'results.json'

    try:
        with open(output_summary, 'r') as fp:
            data = json.load(fp)
    except Exception:
        data = dict()

    with open(output_summary, 'w+') as fp:
        data[dataset] = dict()
        data[dataset]['mean'] = acc
        data[dataset]['std'] = np.std(accs)

        json.dump(data, fp, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config_file', help='configuration file')
    args = parser.parse_args()

    score_predictions(args)
