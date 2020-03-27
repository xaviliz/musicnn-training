import numpy as np
import config_file, shared
from scipy.fftpack import dct
from tqdm import tqdm


def process(x):
    try:
        abs_path = config_file.DATA_FOLDER + config_file.config_train['spec']['audio_representation_folder'] + x + '.npy'
        audio_rep = np.load(open(abs_path, 'rb'))
        d_time = dct(audio_rep, type=2, n=config['discrimintor_dimensions'] + 1, axis=1, norm=None, overwrite_x=False)

        # Remove first coefficient
        d_time = d_time[:, 1:]

        d = np.hstack([np.mean(d_time, axis=0), np.std(d_time, axis=0)])

        return d.min(), d.max()
    
    except:
        print("{} failed".format(abs_path))
        return (0, 0)


if __name__ == '__main__':
    config = config_file.config_train['spec']
    file_index = config_file.DATA_FOLDER + config['audio_representation_folder'] + 'index.tsv'
    [audio_repr_paths, id2audio_repr_path] = shared.load_id2path(file_index)

    # file_ground_truth_train = config_file.DATA_FOLDER + config['gt_train']
    # [ids_train, id2gt_train] = shared.load_id2gt(file_ground_truth_train)
    # print(id2audio_repr_path[ids_train[0]])

    # audio_repr_paths = audio_repr_paths[:10]

    minimum = 0
    maximum = 0

    for i in tqdm(audio_repr_paths):
        lmin, lmax = process(i)

        if lmin < minimum:
            minimum = lmin

        if lmax > maximum:
            maximum = lmax

    print("min: {}".format(minimum))
    print("max: {}".format(maximum))
