from essentia.streaming import MonoLoader, TensorflowInputVGGish, FrameCutter
from essentia import Pool, run

SAMPLE_RATE = 16000
HOP_SIZE = 160
FRAME_SIZE = 400


def feature_melspectrogram_vggish(audio_file):
    pool = Pool()

    loader = MonoLoader(sampleRate=SAMPLE_RATE, filename=audio_file)
    frameCutter = FrameCutter(frameSize=FRAME_SIZE, hopSize=HOP_SIZE)
    mels = TensorflowInputVGGish()

    loader.audio >> frameCutter.signal
    frameCutter.frame >> mels.frame
    mels.bands >> (pool, 'mel_bands')

    run(loader)

    return pool['mel_bands']
