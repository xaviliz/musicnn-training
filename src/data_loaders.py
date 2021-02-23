import numpy as np
import pickle
import random
import shared
import os
import json


def get_audio_rep(config, audio_repr_path, sampling):
    floats_num = os.path.getsize(audio_repr_path) // 2  # each float16 has 2 bytes
    frames_num = floats_num // config['yInput']

    if frames_num < config['xInput']:
        fp = np.memmap(audio_repr_path, dtype='float16', mode='r', shape=(frames_num, config['yInput']))

        audio_rep = np.zeros([config['xInput'], config['yInput']])
        audio_rep[:frames_num, :] = np.array(fp)
        # raise Exception('get_audio_rep: {} contains {} frames, while at least {} are required.'.format(audio_repr_path, frames_num, config['xInput']))
    else:
        if sampling == 'random':
            random_frame_offset = random.randint(0, frames_num - config['xInput'])
            random_offset = random_frame_offset * config['yInput'] * 2  # idx * bands * bytes per float
            fp = np.memmap(audio_repr_path, dtype='float16', mode='r', shape=(config['xInput'], config['yInput']), offset=random_offset)
        elif sampling == 'overlap_sampling':
            fp = np.memmap(audio_repr_path, dtype='float16', mode='r', shape=(frames_num, config['yInput']))

        audio_rep = np.array(fp)
    del fp

    # do not apply any compression to the embeddings
    if config['audio_rep']['type'] == 'embeddings':
        return audio_rep

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
        audio_rep = get_audio_rep(config, audio_repr_path, sampling)

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
