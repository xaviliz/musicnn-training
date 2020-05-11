import os
import numpy as np
import essentia.standard as es
from subprocess import call


# BASE_PATH = '/mnt/mtgdb-audio/stable/'
BASE_PATH = '/home/palonso/data/raw/stable/'

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

KEY_DICT = {
    "A" : [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "Bb": [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "B":  [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "C":  [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    "C#": [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    "D":  [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    "Eb": [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    "E":  [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    "F":  [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    "F#": [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    "G":  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    "Ab": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
}

TONALITY = {
    'minor': [0],
    'major': [1]
}

def set_lowlevel_task(config):
    if config['lowlevel_descriptor'] == 'loudness':
        config['target_descriptor'] = 'lowlevel.loudness_ebu128.integrated'
        config['num_classes_dataset'] = 1
    elif config['lowlevel_descriptor'] == 'bpm':
        config['target_descriptor'] = 'rhythm.bpm'
        config['num_classes_dataset'] = 1
    elif config['lowlevel_descriptor'] == 'hpcp':
        config['target_descriptor'] = 'tonal.hpcp.mean'
        config['num_classes_dataset'] = 36
    elif config['lowlevel_descriptor'] == 'mfcc':
        config['target_descriptor'] = 'lowlevel.mfcc.mean'
        config['num_classes_dataset'] = 14
    elif config['lowlevel_descriptor'] == 'key':
        config['target_descriptor'] = 'tonal.key_krumhansl.key'
        config['num_classes_dataset'] = 12
    elif config['lowlevel_descriptor'] == 'mode':
        config['target_descriptor'] = 'tonal.key_krumhansl.scale'
        config['num_classes_dataset'] = 1
    else:
        raise Exception('lowlevel task not available')

def loudness_encriptor(gt):
    """Loudness binarizer
    """
    l_vect = np.zeros(1)

    """ For the 3-bin case
    if gt < -24.31:
        l_vect[0] = 1
    elif gt < -15.2:
        l_vect[1] = 1
    else:
        l_vect[2] = 1
    """
    if gt > -20.596728802500003:
        l_vect[0] = 1

    return l_vect

def mel_2_mfcc(x, n=24, x_min=-1, x_max=1, headroom=.01):
    """Get MFCC from the mel-bands
    """
    d_time = dct(x, type=2, n=n + 1, axis=1, norm=None, overwrite_x=False)

    # Remove first coefficient
    d_time = d_time[:, 1:]
    d = np.hstack([np.mean(d_time, axis=0), np.std(d_time, axis=0)])
    return minmax_standarize(d, x_max=x_max,
                                x_min=x_min,
                                headroom=headroom)

def get_audio_file(audio_repr_path):
    relative_path = '/'.join(audio_repr_path.split('/')[1:])[:-2]
    dataset_name = audio_repr_path.split('__')[0].split('/')[-1]
    mid_path, ext =  DATASETS_DATA[dataset_name]

    return os.path.join(BASE_PATH, mid_path, relative_path) + ext

def get_loudness(data_folder, audio_repr_path):
    filepath = os.path.join(data_folder, audio_repr_path)
    loudness_file = filepath[:-3] + '_loudness_vect.npy'

    if os.path.exists(loudness_file):
        loudness = np.load(loudness_file)
    else:
        descriptors_file = filepath[:-3] + '_descriptors.json'
        descriptors = get_descripors(audio_repr_path, descriptors_file)
        loudness = loudness_encriptor(descriptors['lowlevel.loudness_ebu128.integrated'])
        np.save(loudness_file, loudness)
    return loudness

def get_bpm(data_folder, audio_repr_path):
    filepath = os.path.join(data_folder, audio_repr_path)
    bpm_file = filepath[:-3] + '_bpm_value.npy'

    if os.path.exists(bpm_file):
        bpm = np.load(bpm_file)
    else:
        descriptors_file = filepath[:-3] + '_descriptors.json'
        descriptors = get_descriptors(audio_repr_path, descriptors_file)
        bpm = descriptors['rhythm.bpm']
        np.save(bpm_file, bpm)

    return bucketize_bpm(bpm)

def bucketize_bpm(bpm):
    """BPM bucketizer
    """
    l_vect = np.zeros(1)
    if bpm > 126.25896072:
        l_vect[0] = 1

    return l_vect

def get_descriptors(audio_repr_path, descriptors_file):
    if os.path.exists(descriptors_file):
        descriptors = es.YamlInput(format='json', filename=descriptors_file)()
    else:
        audio_file = get_audio_file(audio_repr_path)
        descriptors, _ = es.MusicExtractor()(audio_file)
        es.YamlOutput(format='json', filename=descriptors_file)(descriptors)
    return descriptors

def get_key_mode(data_folder, audio_repr_path):
    filepath = os.path.join(data_folder, audio_repr_path)
    audio_file = get_audio_file(audio_repr_path)
    key_file = filepath[:-3] + '_key.npy'

    if os.path.exists(key_file):
        key_vect = np.load(key_file)
    else:
        key_vect = shared.compute_key(audio_file, key_file)

    return key_vect

def get_mode(data_folder, audio_repr_path):
    return np.array([get_key_mode(data_folder, audio_repr_path)[-1]])

def get_key(data_folder, audio_repr_path):
    return get_key_mode(data_folder, audio_repr_path)[:12]

def get_features(data_folder, audio_repr_path):
    filepath = os.path.join(data_folder, audio_repr_path)
    audio_file = get_audio_file(audio_repr_path)
    descriptors_file = filepath[:-3] + '_descriptors.json'

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

    if config['target_descriptor'] == 'lowlevel.loudness_ebu128.integrated':
        gt = loudness_encriptor(gt)
  
    if type(gt) == float:
        gt = np.array([gt])


def get_essentia_spectrogram(config, data_folder, audio_repr_path):
    filename = get_audio_file(audio_repr_path)
    gain = config['audio_effects']['gain']
    rep_filename = filename + str(gain) + '.npy'

    if os.path.exists(rep_filename):
        return np.load(rep_filename)
    else:
        audio = es.MonoLoader(filename=filename, sampleRate=16000)()
        # apply gain
        if gain != 0:
            audio *= 10 ** (gain / 20)

        extractor = es.TensorflowInputMusiCNN()
        rep = np.array([extractor(frame) for frame in es.FrameGenerator(audio, frameSize=512, hopSize=256)])
        np.save(rep_filename, rep)

        return rep

def get_essentia_spectrogram_bpm(config, data_folder, audio_repr_path):
    filename = get_audio_file(audio_repr_path)
    bpm = config['audio_effects']['bpm']
    rep_filename = filename + '_bpm_' + str(bpm) + '.npy'

    if os.path.exists(rep_filename):
        return np.load(rep_filename)
    else:
        altered_filename = filename + '_bpm_' + str(bpm) + '.mp3'

        if not os.path.exists(altered_filename):
            call(['ffmpeg', '-i', filename, '-af', 'atempo={}'.format(bpm), altered_filename])

        audio = es.MonoLoader(filename=altered_filename, sampleRate=16000)()
        
        extractor = es.TensorflowInputMusiCNN()
        rep = np.array([extractor(frame) for frame in es.FrameGenerator(audio, frameSize=512, hopSize=256)])
        np.save(rep_filename, rep)

        return rep

def get_essentia_spectrogram_key(config, data_folder, audio_repr_path):
    filename = get_audio_file(audio_repr_path)
    key = config['audio_effects']['key']
    rep_filename = filename + '_key_' + str(key) + '.npy'

    if os.path.exists(rep_filename):
        return np.load(rep_filename)
    else:
        altered_filename = filename + '_key_' + str(key) + '.mp3'

        if not os.path.exists(altered_filename):
            ratio = 2 ** (int(key) / 12)
            call(['ffmpeg', '-i', filename, '-filter:a', 'rubberband=pitch={}'.format(ratio), altered_filename])

        audio = es.MonoLoader(filename=altered_filename, sampleRate=16000)()
        
        extractor = es.TensorflowInputMusiCNN()
        rep = np.array([extractor(frame) for frame in es.FrameGenerator(audio, frameSize=512, hopSize=256)])
        np.save(rep_filename, rep)

        return rep
