import numpy as np
import pickle
from numpy import random
import shared

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