from essentia.pytools.extractors.melspectrogram import melspectrogram

SR = 16000
N_MELS = 96
FRAME_SIZE = 512
HOP_SIZE = 256


def feature_melspectrogram_essentia(audio_file):
    return melspectrogram(audio_file,
                          sample_rate=SR,
                          frame_size=FRAME_SIZE,
                          hop_size=HOP_SIZE,
                          window_type='hann',
                          low_frequency_bound=0,
                          high_frequency_bound=SR / 2,
                          number_bands=N_MELS,
                          warping_formula='slaneyMel',
                          weighting='linear',
                          normalize='unit_tri',
                          bands_type='power',
                          compression_type='none')
