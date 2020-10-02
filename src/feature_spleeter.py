from essentia.standard import MonoLoader, TensorflowPredict
from essentia import Pool
import numpy as np
from skimage.measure import block_reduce

I_NODE = 'waveform'
O_NODES = ['conv2d_5/BiasAdd', 'conv2d_12/BiasAdd', 'conv2d_19/BiasAdd', 'conv2d_26/BiasAdd', 'conv2d_33/BiasAdd']
SR = 44100
EPS = np.finfo('float32').eps


def feature_spleeter(audio_file):
    audio = MonoLoader(filename=audio_file, sampleRate=SR)()

    stereo = np.vstack([audio, audio]).reshape([-1, 2, 1, 1])

    pool = Pool()
    pool.set(I_NODE, stereo)
    pool = TensorflowPredict(graphFilename='models/spleeter-5s.pb',
                             inputs=[I_NODE], outputs=O_NODES, squeeze=True)(pool)

    feats = []
    for node in O_NODES:
        bn = pool[node]
        # Reorder axes: [batch, time, freq, channels]
        bn = np.swapaxes(bn, 1, 2)
        # Merge batch and time as a single dimension
        bn = np.reshape(bn, [-1, 8, 512])
        # 4 X 4 max pooling along frequencies and channels
        bn = block_reduce(bn, (1, 4, 4))
        # Merge frequency and channel dimensions
        bn = np.reshape(bn, [-1, bn.shape[-1] * bn.shape[-2]])
        feats.append(bn)

    # Stack frequencies and channels of each stem of the model
    embeddings = np.hstack(feats)

    # Apply log10 but keeping the sign information
    return np.sign(embeddings) * np.log10(np.abs(embeddings + EPS))
