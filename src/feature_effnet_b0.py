from essentia.streaming import *
from essentia import Pool, run
import numpy as np
from time import time

def feature_effnet_b0(audio_file):
    # Currently the .pb models converted from Pytorch are
    # an order of magnitude slower than the original
    # implementation. Using the original implementation
    # is strongly recommended until this problem is fixed.
    modelName = 'models/effnetb0_bn200_4n_500k_400l.pb'
    input_layer = 'melspectrogram'
    output_layer = 'add_166'

    # analysis parameters
    sampleRate = 16000
    frameSize=512
    hopSize=256

    # mel bands parameters
    numberBands=96
    weighting='linear'
    warpingFormula='slaneyMel'
    normalize='unit_tri'

    # model parameters
    patchSize = 128

    # Algorithms for mel-spectrogram computation
    audio = MonoLoader(filename=audio_file, sampleRate=sampleRate)
    fc = FrameCutter(frameSize=frameSize, hopSize=hopSize)

    w = Windowing(normalized=False)

    spec = Spectrum()

    mel = MelBands(numberBands=numberBands, sampleRate=sampleRate,
                highFrequencyBound=sampleRate // 2,
                inputSize=frameSize // 2 + 1,
                weighting=weighting, normalize=normalize,
                warpingFormula=warpingFormula)

    # Algorithms for logarithmic compression of mel-spectrograms
    shift = UnaryOperator(shift=1, scale=10000)

    comp = UnaryOperator(type='log10')

    # This algorithm cuts the mel-spectrograms into patches
    # according to the model's input size and stores them in a data
    # type compatible with TensorFlow
    vtt = VectorRealToTensor(shape=[-1, 1, patchSize, numberBands])

    # Auxiliar algorithm to store tensors into pools
    ttp = TensorToPool(namespace=input_layer)

    # The core TensorFlow wrapper algorithm operates on pools
    # to accept a variable number of inputs and outputs
    tfp = TensorflowPredict(graphFilename=modelName,
                            inputs=[input_layer],
                            outputs=[output_layer])

    # Algorithms to retrieve the predictions from the wrapper
    ptt = PoolToTensor(namespace=output_layer)

    ttv = TensorToVectorReal()

    # Another pool to store output predictions
    pool = Pool()

    audio.audio    >>  fc.signal
    fc.frame       >>  w.frame
    w.frame        >>  spec.frame
    spec.spectrum  >>  mel.spectrum
    mel.bands      >>  shift.array
    shift.array    >>  comp.array
    comp.array     >>  vtt.frame
    vtt.tensor     >>  ttp.tensor
    ttp.pool       >>  tfp.poolIn
    tfp.poolOut    >>  ptt.pool
    ptt.tensor     >>  ttv.tensor
    ttv.frame      >>  (pool, output_layer)

    run(audio)

    return pool[output_layer]
