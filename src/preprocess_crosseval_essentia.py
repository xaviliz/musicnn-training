import os
import librosa
from joblib import Parallel, delayed
import json
import argparse
import pickle
import numpy as np
from pathlib import Path
import yaml
from argparse import Namespace
from tqdm import tqdm

config_file = Namespace(**yaml.load(open('config_file.yaml'),
                                    Loader=yaml.SafeLoader))
config = config_file.config_preprocess['mtgdb_spec']

DEBUG = False


def compute_audio_repr(audio_file, audio_repr_file, lib, force=False):
    if not force:
        if os.path.exists(audio_repr_file):
            print('{} exists. skipping!'.format(audio_file))
            return 0

    base_dir = '/'.join(audio_repr_file.split('/')[:-1])
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    if lib == 'essentia':
        from essentiamel import essentia_melspectrogram
        audio_repr = essentia_melspectrogram(audio_file,
                                             sampleRate=config['resample_sr'],
                                             frameSize=config['n_fft'],
                                             hopSize=config['hop'],
                                             warpingFormula='slaneyMel',
                                             window='hann',
                                             normalize='unit_tri',
                                             numberBands=config['n_mels'])
    elif lib == 'librosa':
        import librosa
        audio, sr = librosa.load(audio_file, sr=config['resample_sr'])
        audio_repr = librosa.feature.melspectrogram(y=audio, sr=sr,
                                                    hop_length=config['hop'],
                                                    n_fft=config['n_fft'],
                                                    n_mels=config['n_mels']).T
    else:
        raise Exception('no signal processing lib defined!')

    # Compute length
    print(audio_repr.shape)
    length = audio_repr.shape[0]

    # Transform to float16 (to save storage, and works the same)
    audio_repr = audio_repr.astype(np.float16)

    # Write results:
    with open(audio_repr_file, "wb") as f:
        pickle.dump(audio_repr, f)  # audio_repr shape: NxM

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
        fw = open(os.path.join(data_dir, "index.tsv"), "a")
        fw.write("%s\t%s\t%s\n" % (id, audio_repr_file, audio_file))
        fw.close()
        print(str(index) + '/' + str(len(files)) + ' Computed: %s' % audio_file)

    except Exception as e:
        ferrors = open(os.path.join(data_dir, "index.tsv"), "a")
        ferrors.write(audio_file + "\n")
        ferrors.write(str(e))
        ferrors.close()
        print('Error computing audio representation: ', audio_file)
        print(str(e))


def process_files(files, lib):
    if DEBUG:
        print('WARNING: Parallelization is not used!')
        for index in tqdm(range(0, len(files))):
            do_process(files, index)
    else:
        Parallel(n_jobs=config['num_processing_units'])(
            delayed(do_process)(files, index, lib) for index in range(0, len(files)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('index_file', help='index file')
    parser.add_argument('index_basedir', help='grountruth file')
    parser.add_argument('data_dir', help='grountruth file')
    parser.add_argument('lib', help='dsp lib', choices=['essentia', 'librosa'])

    args = parser.parse_args()

    index_file = args.index_file
    index_basedir = args.index_basedir
    data_dir = args.data_dir
    lib = args.lib

    # set audio representations folder
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    else:
        print("WARNING: already exists a folder with this name!"
              "\nThis is expected if you are splitting computations into different machines.."
              "\n..because all these machines are writing to this folder. Otherwise, check your config_file!")

    # list audios to process: according to 'index_file'
    files_to_convert = []
    f = open(index_file)
    for line in f.readlines():
        id, path = line.strip().split("\t")

        audio_repr = path[:path.rfind(".")] + ".pk"  # .npy or .pk
        audio_repr = audio_repr.replace(index_basedir + '/', '')
        audio_repr = os.path.join(data_dir, audio_repr)
        files_to_convert.append((id, path, audio_repr))

    process_files(files_to_convert, lib)

    # json.dump(config, open(config_file.DATA_FOLDER + config['audio_representation_folder'] + "config.json", "w"))
    # compute audio representation
    # if config['machine_i'] == config['n_machines'] - 1:
    # we just save parameters once! In the last thread run by n_machine-1!
    # else:
    #     first_index = int(len(files_to_convert) / config['n_machines']) * (config['machine_i'])
    #     second_index = int(len(files_to_convert) / config['n_machines']) * (config['machine_i'] + 1)
    #     assigned_files = files_to_convert[first_index:second_index]
    #     process_files(assigned_files)

    # print("Audio representation folder: " + config_file.DATA_FOLDER + config['audio_representation_folder'])
