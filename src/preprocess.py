import argparse
import json
import os
from argparse import Namespace
from pathlib import Path

import librosa
import numpy as np
import yaml
from joblib import Parallel, delayed
from tqdm import tqdm

from feature_effnet_b0 import feature_effnet_b0
from feature_melspectrogram_essentia import feature_melspectrogram_essentia
from feature_melspectrogram_vggish import feature_melspectrogram_vggish
from feature_ol3 import feature_ol3
from feature_spleeter import feature_spleeter
from feature_tempocnn import feature_tempocnn

DEBUG = False


def compute_audio_repr(audio_file, audio_repr_file, force=False):
    if not force:
        if os.path.exists(audio_repr_file):
            print('{} exists. skipping!'.format(audio_repr_file))
            return 0

    if config['config_train']['feature_type'] == 'waveform':
        audio, sr = librosa.load(audio_file, sr=config['resample_sr'])
        audio_repr = audio
        audio_repr = np.expand_dims(audio_repr, axis=1)

    if config['config_train']['feature_type'] == 'musicnn-melspectrogram':
        audio_repr = feature_melspectrogram_essentia(audio_file)
    elif config['config_train']['feature_type'] == 'vggish-melspectrogram':
        audio_repr = feature_melspectrogram_vggish(audio_file)
    elif config['config_train']['feature_type'] == 'ol3':
        audio_repr = feature_ol3(audio_file)
    elif config['config_train']['feature_type'] == 'spleeter':
        audio_repr = feature_spleeter(audio_file)
    elif config['config_train']['feature_type'] == 'tempocnn':
        audio_repr = feature_tempocnn(audio_file)
    elif config['config_train']['feature_type'] == 'effnet_b0':
        audio_repr = feature_effnet_b0(audio_file)
    else:
        raise Exception('Feature {} not implemented.'.format(config['config_train']['feature_type']))

    # Compute length
    length = audio_repr.shape[0]

    # Transform to float16 (to save storage, and works the same)
    audio_repr = audio_repr.astype(np.float16)

    # Write results:
    fp = np.memmap(audio_repr_file, dtype='float16', mode='w+', shape=audio_repr.shape)
    fp[:] = audio_repr[:]
    del fp
    return length

def do_process(files, index):
    try:
        [id, audio_file, audio_repr_file] = files[index]
        if not os.path.exists(audio_repr_file[:audio_repr_file.rfind('/') + 1]):
            path = Path(audio_repr_file[:audio_repr_file.rfind('/') + 1])
            path.mkdir(parents=True, exist_ok=True)
        # compute audio representation (pre-processing)
        length = compute_audio_repr(audio_file, audio_repr_file)
        # index.tsv writing
        fw = open(audio_representation_dir / "index.tsv", "a")
        fw.write("%s\t%s\t%s\n" % (id, audio_repr_file[len(config['data_dir']):], audio_file[len(config['data_dir']):]))
        fw.close()
        print(str(index) + '/' + str(len(files)) + ' Computed: %s' % audio_file)

    except Exception as e:
        ferrors = open(audio_representation_dir / "errors.txt", "a")
        ferrors.write(audio_file + "\n")
        ferrors.write(str(e))
        ferrors.close()
        print('Error computing audio representation: ', audio_file)
        print(str(e))


def process_files(files):
    if DEBUG:
        print('WARNING: Parallelization is not used!')
        for index in tqdm(range(0, len(files))):
            do_process(files, index)

    else:
        Parallel(n_jobs=config['config_preprocess']['num_processing_units'], prefer="threads")(
            delayed(do_process)(files, index) for index in range(0, len(files)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config_file', help='configuration file')
    args = parser.parse_args()

    config = json.load(open(args.config_file, "r"))

    audio_representation_dir = Path(config['config_train']['audio_representation_dir'])

    # set audio representations folder
    audio_representation_dir.mkdir(parents=True, exist_ok=True)

    # list audios to process: according to 'index_file'
    files_to_convert = []
    f = open(Path(config['data_dir'], config['config_preprocess']['index_audio_file']))
    for line in f.readlines():
        id, audio = line.strip().split("\t")
        audio_repr = audio[:audio.rfind(".")] + ".dat" # .npy or .pk
        files_to_convert.append((id, str(Path(config['config_preprocess']['audio_dir'], audio)),
                                 str(audio_representation_dir / audio_repr)))

    # compute audio representation
    process_files(files_to_convert)

    print("Audio representation folder: ", audio_representation_dir)
