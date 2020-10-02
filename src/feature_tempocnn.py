from essentia.standard import MonoLoader, TensorflowPredictTempoCNN
from essentia import Pool, run
import numpy as np

PATCH_HOPSIZE = 16
SR = 11025
MIN_TIME = 30


def feature_tempocnn(audio_file):
    audio = MonoLoader(filename=audio_file, sampleRate=SR)()

    if len(audio) < SR * MIN_TIME:
        padding = SR * MIN_TIME - len(audio)
        r_padding = np.zeros(padding // 2, dtype='float32')
        l_padding = np.zeros(padding - len(r_padding), dtype='float32')
        audio = np.hstack([l_padding, audio, r_padding])

    return TensorflowPredictTempoCNN(graphFilename='models/deepsquare_k16.pb',
                                     output='1x1/Relu0_reshape',
                                     patchHopSize=PATCH_HOPSIZE)(audio)
