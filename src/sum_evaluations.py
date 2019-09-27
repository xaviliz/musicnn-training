import argparse
import json
import pescador
import shared, train
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import yaml
from argparse import Namespace

config_file = Namespace(**yaml.load(open('config_file.yaml')))


TEST_BATCH_SIZE = 64
FILE_INDEX = config_file.DATA_FOLDER + 'index_repr.tsv'
FILE_GROUND_TRUTH_TEST = config_file.config_train['spec']['gt_test']
FOLD = config_file.config_train['spec']['fold']


if __name__ == '__main__':

    data_folder = config_file.DATA_FOLDER

    n_folds = config_file.config_train['spec']['n_folds']

    results = dict()
    for fold in range(n_folds):
        with open(data_folder + 'predictions_{}.json'.format(fold)) as f:
            data = json.load(f)
            results.update({data})

        # except:
        #     print('failed to load predictions_{}.json'.format(fold))

    print(results)
