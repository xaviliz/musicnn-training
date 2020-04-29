import numpy as np
import pickle
from numpy import random
import shared
import os
import json
import essentia.standard as es
from shared import KEY_DICT, TONALITY


BASE_PATH = '/mnt/mtgdb-audio/stable/'

DATASETS_DATA = {
    'ismir04_rhythm':     ['ismir04_rhythm/'],
    'moods_mirex':        ['moods_mirex/'],
    'genre_dortmund':     ['genre_dortmund/audio/wav/', 'wav'],
    'genre_tzanetakis':   ['genre_tzanetakis/audio/22kmono/', 'wav'],
    'genre_electronic':   ['genre_electronic/audio/wav/', 'wav'],
    'genre_rosamerica':   ['genre_rosamerica/audio/mp3/', 'mp3'],
    'voice_instrumental': ['voice_instrumental/audio/mp3/', 'mp3'],
    'tonal_atonal':       ['tonal_atonal/audio/mp3/', 'MP3'],
    'timbre':             ['timbre_bright_dark/audio/mp3/', 'MP3'],
    'gender':             ['gender/audio/mp3', 'MP3'],
    'danceability':       ['danceability/audio/mp3/', 'mp3'],
    'mood_acoustic':      ['moods_claurier/audio/mp3/', 'mp3'],
    'mood_aggressive':    ['moods_claurier/audio/mp3/', 'mp3'],
    'mood_electronic':    ['moods_claurier/audio/mp3/', 'mp3'],
    'mood_happy':         ['moods_claurier/audio/mp3/', 'mp3'],
    'mood_party':         ['moods_claurier/audio/mp3/', 'mp3'],
    'mood_relaxed':       ['moods_claurier/audio/mp3/', 'mp3'],
    'mood_sad':           ['moods_claurier/audio/mp3/', 'mp3']
    }


def data_gen_standard(id, audio_repr_path, gt, pack):
    [config, sampling, param_sampling, augmentation, data_folder] = pack
    audio_repr_path = data_folder + audio_repr_path

    # load audio representation -> audio_repr shape: NxM
    audio_rep = pickle.load(open(audio_repr_path, 'rb'))
    if config['pre_processing'] == 'logEPS':
        audio_rep = np.log10(audio_rep + np.finfo(float).eps)
    elif  config['pre_processing'] == 'logC':
        audio_rep = np.log10(10000 * audio_rep + 1)

    # let's deliver some data!
    last_frame = int(audio_rep.shape[0]) - int(config['xInput']) + 1
    if sampling == 'random':
        for i in range(0, param_sampling):
            time_stamp = random.randint(0,last_frame-1)
            yield dict(X = audio_rep[time_stamp : time_stamp+config['xInput'], : ], Y = gt, ID = id)

    elif sampling == 'overlap_sampling':
        for time_stamp in range(0, last_frame, param_sampling):
            yield dict(X = audio_rep[time_stamp : time_stamp+config['xInput'], : ], Y = gt, ID = id)

def data_gen_standard_abs_path(id, audio_repr_path, gt, pack):

    [config, sampling, param_sampling, augmentation] = pack

    # load audio representation -> audio_repr shape: NxM
    audio_rep = pickle.load(open(audio_repr_path, 'rb'))
    if config['pre_processing'] == 'logEPS':
        audio_rep = np.log10(audio_rep + np.finfo(float).eps)
    elif  config['pre_processing'] == 'logC':
        audio_rep = np.log10(10000 * audio_rep + 1)

    # let's deliver some data!
    last_frame = int(audio_rep.shape[0]) - int(config['xInput']) + 1
    if sampling == 'random':
        for i in range(0, param_sampling):
            time_stamp = random.randint(0,last_frame-1)
            yield dict(X = audio_rep[time_stamp : time_stamp+config['xInput'], : ], Y = gt, ID = id)

    elif sampling == 'overlap_sampling':
        for time_stamp in range(0, last_frame, param_sampling):
            yield dict(X = audio_rep[time_stamp : time_stamp+config['xInput'], : ], Y = gt, ID = id)


def data_gen_discriminator(id, audio_repr_path, gt, pack):
    [config, sampling, param_sampling, augmentation, data_folder] = pack
    abs_path = data_folder + audio_repr_path

    # load audio representation -> audio_repr shape: NxM
    try:
        audio_rep = np.load(open(abs_path, 'rb'), allow_pickle=True)
        # audio_rep = pickle.load(open(abs_path, 'rb'))
        if config['pre_processing'] == 'logEPS':
            audio_rep = np.log10(audio_rep + np.finfo(float).eps)
        elif  config['pre_processing'] == 'logC':
            audio_rep = np.log10(10000 * audio_rep + 1)

        # let's deliver some data!
        last_frame = int(audio_rep.shape[0]) - int(config['xInput']) + 1
        if sampling == 'random':
            time_stamp = random.randint(0,last_frame-1)
            x = audio_rep[time_stamp : time_stamp+config['xInput'], :]

            d = np.clip(shared.mel_2_mfcc(x, n=config['discriminator_dimensions'],
                                            x_min=config['mfcc_min'],
                                            x_max=config['mfcc_max'],
                                            headroom=config['mfcc_headroom']), 0, 1)
            yield dict(X = x, Y = gt, ID = id, D = d)

        elif sampling == 'overlap_sampling':
            for time_stamp in range(0, last_frame, param_sampling):
                x = audio_rep[time_stamp : time_stamp+config['xInput'], : ]

                d = np.clip(shared.mel_2_mfcc(x, n=config['discriminator_dimensions'],
                                                x_min=config['mfcc_min'],
                                                x_max=config['mfcc_max'],
                                                headroom=config['mfcc_headroom']), 0, 1)

                yield dict(X = x, Y = gt, ID = id, D = d)
    except:
        print('"{}" failed'.format(abs_path))


def data_gen_key(id, audio_repr_path, gt, pack):
    [config, sampling, param_sampling, augmentation, data_folder] = pack
    abs_path = data_folder + audio_repr_path

    key_file = abs_path[:-3] + '_key'

    if os.path.exists(key_file + '.npy'):
        key_vect = np.load(key_file + '.npy')
    else:
        relative_path = '/'.join(audio_repr_path.split('/')[1:])[:-2]
        dataset_name = audio_repr_path.split('__')[0]
        mid_path, ext =  DATASETS_DATA[dataset_name]
        audio_file = os.path.join(BASE_PATH,mid_path, relative_path) + ext
        try:
            key_vect = shared.compute_key(audio_file, key_file)
        except:
            print('audio_file: {}'.format(audio_file))
            print('key_file: {}'.format(key_file))
            print('failed to compute key of {}'.format(audio_repr_path))

    if config['discriminator_target'] == 'key_mode':
        pass
    elif config['discriminator_target'] == 'key':
        key_vect = key_vect[:-1]
    elif config['discriminator_target'] == 'mode':
        key_vect = np.array([key_vect[-1], key_vect[-1]])

    # load audio representation -> audio_repr shape: NxM
    try:
        audio_rep = np.load(open(abs_path, 'rb'), allow_pickle=True)
        # audio_rep = pickle.load(open(abs_path, 'rb'))
        if config['pre_processing'] == 'logEPS':
            audio_rep = np.log10(audio_rep + np.finfo(float).eps)
        elif  config['pre_processing'] == 'logC':
            audio_rep = np.log10(10000 * audio_rep + 1)

        # let's deliver some data!
        last_frame = int(audio_rep.shape[0]) - int(config['xInput']) + 1
        if sampling == 'random':
            time_stamp = random.randint(0,last_frame-1)
            x = audio_rep[time_stamp : time_stamp+config['xInput'], :]

            yield dict(X = x, Y = gt, ID = id, D = key_vect)

        elif sampling == 'overlap_sampling':
            for time_stamp in range(0, last_frame, param_sampling):
                x = audio_rep[time_stamp : time_stamp+config['xInput'], : ]

                yield dict(X = x, Y = gt, ID = id, D = key_vect)
    except Exception as ex:
        print('"{}" failed'.format(abs_path))
        print(repr(ex))


def data_gen_random(id, audio_repr_path, gt, pack):
    [config, sampling, param_sampling, augmentation, data_folder] = pack
    abs_path = data_folder + audio_repr_path

    disc_vect = [np.random.uniform()]

    # load audio representation -> audio_repr shape: NxM
    try:
        audio_rep = np.load(open(abs_path, 'rb'), allow_pickle=True)
        # audio_rep = pickle.load(open(abs_path, 'rb'))
        if config['pre_processing'] == 'logEPS':
            audio_rep = np.log10(audio_rep + np.finfo(float).eps)
        elif  config['pre_processing'] == 'logC':
            audio_rep = np.log10(10000 * audio_rep + 1)

        # let's deliver some data!
        last_frame = int(audio_rep.shape[0]) - int(config['xInput']) + 1
        if sampling == 'random':
            time_stamp = random.randint(0,last_frame-1)
            x = audio_rep[time_stamp : time_stamp+config['xInput'], :]

            yield dict(X = x, Y = gt, ID = id, D = disc_vect)

        elif sampling == 'overlap_sampling':
            for time_stamp in range(0, last_frame, param_sampling):
                x = audio_rep[time_stamp : time_stamp+config['xInput'], : ]

                yield dict(X = x, Y = gt, ID = id, D = disc_vect)
    except Exception as ex:
        print('"{}" failed'.format(abs_path))
        print(repr(ex))


def data_gen_music_feature(id, audio_repr_path, gt, pack):
    [config, sampling, param_sampling, augmentation, data_folder] = pack
    abs_path = data_folder + audio_repr_path
    descriptors_file = abs_path[:-3] + '_descriptors.json'
    relative_path = '/'.join(audio_repr_path.split('/')[1:])[:-2]
    dataset_name = audio_repr_path.split('__')[0]
    mid_path, ext =  DATASETS_DATA[dataset_name]
    audio_file = os.path.join(BASE_PATH,mid_path, relative_path) + ext

    try:
        if os.path.exists(descriptors_file):
            descriptors = es.YamlInput(format='json', filename=descriptors_file)()
        else:
            descriptors, _ = es.MusicExtractor()(audio_file)
            es.YamlOutput(format='json', filename=descriptors_file)(descriptors)

        gt = descriptors[config['target_descriptor']]

        if config['target_descriptor'] == 'tonal.key_krumhansl.key':
            gt = KEY_DICT[gt]

        if config['target_descriptor'] == 'tonal.key_krumhansl.scale':
            gt = TONALITY[gt]
      
        if type(gt) == float:
            gt = np.array([gt])

        # load audio representation -> audio_repr shape: NxM
        audio_rep = pickle.load(open(abs_path, 'rb'))
        if config['pre_processing'] == 'logEPS':
            audio_rep = np.log10(audio_rep + np.finfo(float).eps)
        elif  config['pre_processing'] == 'logC':
            audio_rep = np.log10(10000 * audio_rep + 1)

        # let's deliver some data!
        last_frame = int(audio_rep.shape[0]) - int(config['xInput']) + 1
        if sampling == 'random':
            for i in range(0, param_sampling):
                time_stamp = random.randint(0,last_frame-1)
                yield dict(X = audio_rep[time_stamp : time_stamp+config['xInput'], : ], Y = gt, ID = id)

        elif sampling == 'overlap_sampling':
            for time_stamp in range(0, last_frame, param_sampling):
                yield dict(X = audio_rep[time_stamp : time_stamp+config['xInput'], : ], Y = gt, ID = id)

    except Exception as e:
        print('audio_file: {}'.format(audio_file))
        print('descriptors_file: {}'.format(descriptors_file))
        print('melbands_file: {}'.format(abs_path))
        print(e)