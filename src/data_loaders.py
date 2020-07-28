import numpy as np
import pickle
from numpy import random
import shared
import os
import json
from feature_functions import *


def get_lowlevel_groundtruth(config, data_folder, audio_repr_path):
    if config['lowlevel_descriptor'] == 'bpm':
        gt = get_bpm(data_folder, audio_repr_path)
    elif config['lowlevel_descriptor'] == 'loudness':
        gt = get_loudness(data_folder, audio_repr_path)
    elif config['lowlevel_descriptor'] == 'key_mode':
        gt = get_key_mode(data_folder, audio_repr_path)
    elif config['lowlevel_descriptor'] == 'key':
        gt = get_key(data_folder, audio_repr_path)
    elif config['lowlevel_descriptor'] == 'mode':
        gt = get_mode(data_folder, audio_repr_path)
    else:
        raise Exception('lowlevel_descriptor not avaiable')
    return gt

def get_degradated_audio_rep(config, data_folder, audio_repr_path):
    if config['alteration'] == 'loudness':
        audio_rep = get_spectrogram_alterated_loudness(config, data_folder, audio_repr_path)
    elif config['alteration'] == 'bpm':
        audio_rep = get_spectrogram_alterated_bpm(config, data_folder, audio_repr_path)
    elif config['alteration'] == 'key':
        audio_rep = get_spectrogram_alterated_key(config, data_folder, audio_repr_path)
    else:
        raise Exception('eval_mode not avaiable')
    return audio_rep

def get_audio_rep(config, audio_repr_path):
    audio_rep = np.load(open(audio_repr_path, 'rb'), allow_pickle=True)
    if config['pre_processing'] == 'logEPS':
        return np.log10(audio_rep + np.finfo(float).eps)
    elif  config['pre_processing'] == 'logC':
        return np.log10(10000 * audio_rep + 1)
    elif  config['pre_processing'] == '':
        return audio_rep
    else:
        raise('get_audio_rep: Preprocessing not available.')

def data_gen_standard(id, relative_audio_repr_path, gt, pack):
    # Support both the absolute and relative path input cases
    if len(pack) == 5:
        [config, sampling, param_sampling, augmentation, data_folder] = pack
        audio_repr_path = data_folder + relative_audio_repr_path
    else:
        [config, sampling, param_sampling, augmentation] = pack
        audio_repr_path =relative_audio_repr_path

    try:
        # Change groundtruth for the lowlevel_descriptors case 
        if config['task'] == 'lowlevel_descriptors':
            gt = get_lowlevel_groundtruth(config, data_folder, audio_repr_path)

        # load audio representation -> audio_repr shape: NxM
        if config['task'] == 'labels':
            audio_rep = get_audio_rep(config, audio_repr_path)
        elif config['task'] == 'lowlevel_descriptors':
            audio_rep = get_audio_rep(config, audio_repr_path)
        elif config['task'] == 'alterations':
            audio_rep = get_degradated_audio_rep(config, data_folder, relative_audio_repr_path)
        else:
            raise Exception('data_loaders: Case not contemplated')

        # let's deliver some data!
        last_frame = int(audio_rep.shape[0]) - int(config['xInput']) + 1
        if sampling == 'random':
            for i in range(0, param_sampling):
                time_stamp = random.randint(0,last_frame-1)
                yield dict(X = audio_rep[time_stamp : time_stamp+config['xInput'], : ], Y = gt, ID = id)

        elif sampling == 'overlap_sampling':
            for time_stamp in range(0, last_frame, param_sampling):
                yield dict(X = audio_rep[time_stamp : time_stamp+config['xInput'], : ], Y = gt, ID = id)
    except Exception as ex:
        print('"{}" failed'.format(audio_repr_path))
        print(repr(ex))

def data_gen_discriminator(id, audio_repr_path, gt, pack):
    [config, sampling, param_sampling, augmentation, data_folder] = pack
    if not '/home/' in audio_repr_path:
        # it's a relative path
        audio_repr_path = data_folder + audio_repr_path

    try:
        if config['discriminator_target'] == 'random':
            d = np.array(np.random.randint(2))
        elif config['discriminator_target'] == 'loudness':
            d = get_loudness(data_folder, audio_repr_path)
        elif config['discriminator_target'] == 'bpm':
            d = get_bpm(data_folder, audio_repr_path)
        elif config['discriminator_target'] == 'key_mode':
            d = get_key_mode(data_folder, audio_repr_path)
        elif config['discriminator_target'] == 'key':
            d = get_key_mode(data_folder, audio_repr_path)[:-1]
        elif config['discriminator_target'] == 'mode':
            d = np.array(get_key_mode(data_folder, audio_repr_path)[-1])
        elif config['discriminator_target'] == 'mfcc':
            pass
        else:
          raise Exception('discriminator_target not available')

        # load audio representation -> audio_repr shape: NxM
        audio_rep = get_audio_rep(config, audio_repr_path)

        # let's deliver some data!
        last_frame = int(audio_rep.shape[0]) - int(config['xInput']) + 1
        if sampling == 'random':
            time_stamp = random.randint(0,last_frame-1)
            x = audio_rep[time_stamp : time_stamp+config['xInput'], :]

            if config['discriminator_target'] == 'mfcc':
                d = np.clip(mel_2_mfcc(x, n=config['discriminator_dimensions'],
                      x_min=config['mfcc_min'],
                      x_max=config['mfcc_max'],
                      headroom=config['mfcc_headroom']), 0, 1)

            yield dict(X = x, Y = gt, ID = id, D = d)

        elif sampling == 'overlap_sampling':
            for time_stamp in range(0, last_frame, param_sampling):
                x = audio_rep[time_stamp : time_stamp+config['xInput'], : ]

                if config['discriminator_target'] == 'mfcc':
                    d = np.clip(mel_2_mfcc(x, n=config['discriminator_dimensions'],
                        x_min=config['mfcc_min'],
                        x_max=config['mfcc_max'],
                        headroom=config['mfcc_headroom']), 0, 1)

                yield dict(X = x, Y = gt, ID = id, D = d)
    except Exception as ex:
        print('"{}" failed'.format(audio_repr_path))
        print(repr(ex))
