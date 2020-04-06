#!/bin/bash

interpreter=python

data_dir=/home/palonso/data/ccetl/librosa
exp_dir=/home/palonso/models/essentia_embeddings/tl/musicnn_msd_late_adverasial_task_projeced_slow_typeb/

model_dir=/home/palonso/reps/mtgdb-models/src/musicnn_training/weights/MSD_musicnn/
source ~/venvs/py3.6_musicnn/bin/activate

${interpreter} -u train.py spec
