import os
from joblib import Parallel, delayed
import json
import argparse
import numpy as np
from pathlib import Path
import yaml
from argparse import Namespace
from tqdm import tqdm

from feature_melspectrogram_essentia import feature_melspectrogram_essentia
from feature_melspectrogram_vggish import feature_melspectrogram_vggish
from feature_ol3 import feature_ol3
from feature_spleeter import feature_spleeter
from feature_tempocnn import feature_tempocnn

config_file = Namespace(**yaml.load(open('config_file.yaml'), Loader=yaml.SafeLoader))

DEBUG = False


def compute_audio_repr(audio_file, audio_repr_file, force=False):
    if not force:
        if os.path.exists(audio_repr_file):
            print('{} exists. skipping!'.format(audio_file))
            return 0

    if config['type'] == 'waveform':
        audio, sr = librosa.load(audio_file, sr=config['resample_sr'])
        audio_repr = audio
        audio_repr = np.expand_dims(audio_repr, axis=1)

    elif config['feature_name'] == 'melspectrogram':
        audio_repr = feature_melspectrogram_essentia(audio_file)
    elif config['feature_name'] == 'vggish':
        audio_repr = feature_melspectrogram_vggish(audio_file)
    elif config['feature_name'] == 'ol3':
        audio_repr = feature_ol3(audio_file)
    elif config['feature_name'] == 'spleeter':
        audio_repr = feature_spleeter(audio_file)
    elif config['feature_name'] == 'tempocnn':
        audio_repr = feature_tempocnn(audio_file)
    else:
        raise Exception('Feature {} not implemented.'.format(config['type']))

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
        fw = open(audio_representation_folder + "index_" + str(config['machine_i']) + ".tsv", "a")
        fw.write("%s\t%s\t%s\n" % (id, audio_repr_file[len(config_file.DATA_FOLDER):], audio_file[len(config_file.DATA_FOLDER):]))
        fw.close()
        print(str(index) + '/' + str(len(files)) + ' Computed: %s' % audio_file)

    except Exception as e:
        ferrors = open(audio_representation_folder + "errors" + str(config['machine_i']) + ".txt", "a")
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
        Parallel(n_jobs=config['num_processing_units'], prefer="threads")(
            delayed(do_process)(files, index) for index in range(0, len(files)))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('feature_name', help='the feature type')
    args = parser.parse_args()
    config = config_file.config_preprocess
    feature_name = args.feature_name
    config.update(config_file.config_preprocess[feature_name])
    config['feature_name'] = feature_name

    audio_representation_folder = config_file.config_train['audio_representation_folder']

    # set audio representations folder
    if not os.path.exists(audio_representation_folder):
        os.makedirs(audio_representation_folder)
    else:
        print("WARNING: already exists a folder with this name!"
              "\nThis is expected if you are splitting computations into different machines.."
              "\n..because all these machines are writing to this folder. Otherwise, check your config_file!")

    # list audios to process: according to 'index_file'
    files_to_convert = []
    f = open(config_file.DATA_FOLDER + config["index_audio_file"])
    for line in f.readlines():
        id, audio = line.strip().split("\t")
        audio_repr = audio[:audio.rfind(".")] + ".dat" # .npy or .pk
        files_to_convert.append((id, config['audio_folder'] + audio,
                                 audio_representation_folder + audio_repr))

    # compute audio representation
    if config['machine_i'] == config['n_machines'] - 1:
        process_files(files_to_convert[int(len(files_to_convert) / config['n_machines']) * (config['machine_i']):])
        # we just save parameters once! In the last thread run by n_machine-1!
        json.dump(config, open(audio_representation_folder + "config.json", "w"))
    else:
        first_index = int(len(files_to_convert) / config['n_machines']) * (config['machine_i'])
        second_index = int(len(files_to_convert) / config['n_machines']) * (config['machine_i'] + 1)
        assigned_files = files_to_convert[first_index:second_index]
        process_files(assigned_files)

    print("Audio representation folder: " + audio_representation_folder)
