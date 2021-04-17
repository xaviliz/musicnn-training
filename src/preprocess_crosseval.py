import os
from joblib import Parallel, delayed
import argparse
import pickle
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

DEBUG = True


def compute_audio_repr(audio_file, audio_repr_file, lib, force=False):
    if not force:
        if os.path.exists(audio_repr_file):
            print('{} exists. skipping!'.format(audio_repr_file))
            return 0

    base_dir = '/'.join(audio_repr_file.split('/')[:-1])
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    if lib == 'essentia':
        audio_repr = feature_melspectrogram_essentia(audio_file)
    elif lib == 'vggish':
        audio_repr = feature_melspectrogram_vggish(audio_file)
    elif lib == 'ol3':
        audio_repr = feature_ol3(audio_file)
    elif lib == 'spleeter':
        audio_repr = feature_spleeter(audio_file)
    elif lib == 'tempocnn':
        audio_repr = feature_tempocnn(audio_file)
    else:
        raise Exception('no signal processing lib defined!')

    # Compute length
    length = audio_repr.shape[0]

    # Transform to float16 (to save storage, and works the same)
    audio_repr = audio_repr.astype(np.float16)

    # Write results:
    fp = np.memmap(audio_repr_file, dtype='float16', mode='w+', shape=audio_repr.shape)
    fp[:] = audio_repr[:]
    del fp

    return length


def do_process(files, index, lib):
    try:
        [id, audio_file, audio_repr_file] = files[index]
        if not os.path.exists(audio_repr_file[:audio_repr_file.rfind('/') + 1]):
            path = Path(audio_repr_file[:audio_repr_file.rfind('/') + 1])
            path.mkdir(parents=True, exist_ok=True)
        # compute audio representation (pre-processing)
        length = compute_audio_repr(audio_file, audio_repr_file, lib)
        # index.tsv writing
        fw = open(os.path.join(metadata_dir, 'index.tsv'), 'a')
        fw.write("%s\t%s\n" % (id, audio_repr_file))
        fw.close()

    except Exception as e:
        print('Error computing audio representation: ', audio_repr_file)
        print('Source file: ', audio_file)
        print(str(e))


def process_files(files, lib):
    if DEBUG:
        print('WARNING: Parallelization is not used!')
        for index in tqdm(range(0, len(files))):
            do_process(files, index, lib)
    else:
        Parallel(n_jobs=config['num_processing_units'])(
            delayed(do_process)(files, index, lib) for index in range(0, len(files)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('index_file', help='index file')
    parser.add_argument('audio_dir', help='grountruth file')
    parser.add_argument('data_dir', help='grountruth file')
    parser.add_argument('lib', help='dsp lib', choices=['essentia', 'librosa', 'vggish', 'ol3', 'spleeter', 'tempocnn'])

    args = parser.parse_args()

    index_file = args.index_file
    audio_dir = args.audio_dir
    data_dir = args.data_dir
    lib = args.lib


    # set audio representations folder
    metadata_dir = os.path.join(data_dir, 'metadata')
    if not os.path.exists(metadata_dir):
        os.makedirs(metadata_dir, exist_ok=True)
    else:
        print("WARNING: already exists a folder with this name!"
              "\nThis is expected if you are splitting computations into different machines.."
              "\n..because all these machines are writing to this folder. Otherwise, check your config_file!")

    fw = open(os.path.join(metadata_dir, 'index.tsv'), "w")
    fw.write('')
    fw.close()

    # list audios to process: according to 'index_file'
    files_to_convert = []
    f = open(index_file)
    for line in f.readlines():
        id, audio_path = line.strip().split("\t")
        audio_repr = audio_path[:audio_path.rfind(".")] + ".dat"
        tgt = os.path.join(data_dir, audio_repr)
        src = os.path.join(audio_dir, audio_path)

        files_to_convert.append((id, src, tgt))

    process_files(files_to_convert, lib)
