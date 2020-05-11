import numpy as np
from datetime import datetime
from sklearn import metrics
import warnings
warnings.filterwarnings('ignore')
from scipy.fftpack import dct
import essentia.standard as es


def get_epoch_time():
    return int((datetime.now() - datetime(1970,1,1)).total_seconds())


def count_params(trainable_variables):
    # to return number of trainable variables. Example: shared.count_params(tf.trainable_variables()))
    return np.sum([np.prod(v.get_shape().as_list()) for v in trainable_variables])


def load_id2gt(gt_file):
    ids = []
    fgt = open(gt_file)
    id2gt = dict()
    for line in fgt.readlines():
        id, gt = line.strip().split("\t") # id is string
        id2gt[id] = eval(gt) # gt is array
        ids.append(id)
    return ids, id2gt


def load_id2path(index_file):
    paths = []
    fspec = open(index_file)
    id2path = dict()
    for line in fspec.readlines():
        id, path, = line.strip().split("\t")
        id2path[id] = path
        paths.append(path)
    return paths, id2path


def auc_with_aggergated_predictions(y_true, y_pred):
    print('now computing AUC..')
    roc_auc, pr_auc = compute_auc(y_true, y_pred)
    return np.mean(roc_auc), np.mean(pr_auc)


def compute_auc(true, estimated):
    pr_auc = []
    roc_auc = []

    estimated = np.array(estimated)
    true = np.array(true)

    for count in range(0, estimated.shape[1]):
        try:
            pr_auc.append(metrics.average_precision_score(true[:,count],estimated[:,count]))
            roc_auc.append(metrics.roc_auc_score(true[:,count],estimated[:,count]))
        except:
            print('failed!')
    return roc_auc, pr_auc

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def minmax_standarize(x, x_min=-1, x_max=1, headroom=.1):
    return (x - x_min) / ((x_max + headroom) - (x_min - headroom))


def average_predictions(pred_array, id_array, ids, id2gt=None):
    # averaging probabilities -> one could also do majority voting
    print('Averaging predictions')
    y_pred = []
    y_true = []
    for id in ids:
        try:
            avg = np.mean(pred_array[np.where(id_array == id)], axis=0)
            if np.isnan(avg).any():
                print('{} skipped because it contains nans'.format(id))
                continue

            if np.isposinf(avg).any():
                print('{} skipped because it contains pos infs'.format(id))
                continue

            if np.isneginf(avg).any():
                print('{} skipped because it contains neg infs'.format(id))
                continue
            y_pred.append(avg)
            if id2gt:
                y_true.append(id2gt[id])
        except:
            print(id)

    if id2gt:
        return y_true, y_pred
    else:
        return y_pred

def compute_accuracy(y_true, y_pred):
    print('computing accuracy of {} elements'.format(len(y_true)))

    if len(y_true[0]) > 1:
        y_true = np.argmax(y_true, axis=1)
        y_pred = np.argmax(y_pred, axis=1)
    else:
        y_true = np.squeeze(y_true)
        y_pred = np.round(np.squeeze(y_pred))

    return metrics.balanced_accuracy_score(y_true, y_pred)

def compute_key(audio_file, key_file):
    audio = es.MonoLoader(filename=audio_file)()
    key, scale, strength = es.KeyExtractor()(audio)

    key_vect = np.array(KEY_DICT[key] + TONALITY[scale])

    np.save(key_file, key_vect)

    return key_vect