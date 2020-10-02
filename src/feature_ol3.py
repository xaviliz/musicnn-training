import essentia.standard as es
from essentia import Pool
import numpy as np

N_DFT = 2048
N_MELS = 128
N_HOP = 242
SR = 48000.
HOP = 12000
PATCH_SIZE = 199
AUDIO_BATCH_SIZE = int(1 * SR)
BATCH_HOP_SIZE = AUDIO_BATCH_SIZE // 4 # make it a param?
PAD_SIZE = 982
FORMULA = 'htkMel'
MEL_TYPE = 'magnitude'
WEIGHTING = 'linear'
AMIN = 1e-10
DRANGE = 80
INPUT = 'melspectrogram'
OUTPUT = 'embeddings'


def feature_ol3(audio_file):
    w = es.Windowing(size=N_DFT,
                    normalized=False,
                    zeroPhase=False)
    s = es.Spectrum(size=N_DFT)
    mb = es.MelBands(highFrequencyBound=SR / 2,
                    inputSize=N_DFT // 2 + 1,
                    log=False,
                    lowFrequencyBound=0,
                    normalize='unit_sum',
                    numberBands=N_MELS,
                    sampleRate=SR,
                    type=MEL_TYPE,
                    warpingFormula=FORMULA,
                    weighting=WEIGHTING)
    tfp = es.TensorflowPredict(graphFilename='models/openl3_audio_mel128_music_emb512.pb',
                            inputs=[INPUT], outputs=[OUTPUT], squeeze=False)
    pool = Pool()

    padding = np.zeros(PAD_SIZE, dtype='float32')

    audio = es.MonoLoader(filename=audio_file, sampleRate=SR)()

    embeddings = []
    for audio_chunk in es.FrameGenerator(audio, frameSize=AUDIO_BATCH_SIZE,
                                         hopSize=HOP):
        audio_chunk_padded = np.hstack([padding, audio_chunk, padding])

        melbands = np.array([mb(s(w(frame))**2) for frame in es.FrameGenerator(audio_chunk_padded,
                    frameSize=N_DFT, hopSize=N_HOP, startFromZero=True)]).T.copy()

        melbands = melbands ** .5
        melbands = np.clip(melbands, AMIN, None)
        melbands = 10. * np.log10(melbands)
        melbands = melbands - np.max(melbands)
        melbands = np.clip(melbands, -DRANGE, None)
        batch = np.expand_dims(melbands, (0, 3))

        pool.set(INPUT, batch)
        embeddings.append(tfp(pool)[OUTPUT].squeeze())

    return np.vstack(embeddings)
