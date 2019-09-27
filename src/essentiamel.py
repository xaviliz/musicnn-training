import essentia.streaming as es
from essentia import Pool, run
import numpy as np


def essentia_melspectrogram(filename, sampleRate=44100, frameSize=2048, hopSize=1024,
                            window='blackmanharris62', zeroPadding=0, center=True,
                            numberBands=96, lowFrequencyBound=0,
                            highFrequencyBound=None, weighting='linear',
                            warpingFormula='slaneyMel', normalize='unit_tri'):

    if highFrequencyBound is None:
        highFrequencyBound = sampleRate / 2

    loader = es.MonoLoader(filename=filename, sampleRate=sampleRate)

    frameCutter = es.FrameCutter(frameSize=frameSize, hopSize=hopSize,
                                 startFromZero=not center)

    windowing = es.Windowing(type=window, normalized=False,
                             zeroPadding=zeroPadding)
    spectrum = es.Spectrum()

    melBands = es.MelBands(numberBands=numberBands,
                           sampleRate=sampleRate,
                           lowFrequencyBound=lowFrequencyBound,
                           highFrequencyBound=highFrequencyBound,
                           inputSize=(frameSize + zeroPadding) // 2 + 1,
                           weighting=weighting,
                           normalize=normalize,
                           warpingFormula=warpingFormula,
                           type='power')

    results = Pool()

    # norm10k = es.UnaryOperator(type='identity', shift=1, scale=10000)
    # log10 = es.UnaryOperator(type='log10')
    # amp2db = es.UnaryOperator(type='lin2db', scale=2)

    loader.audio >> frameCutter.signal
    frameCutter.frame >> windowing.frame >> spectrum.frame
    spectrum.spectrum >> melBands.spectrum
    melBands.bands >> (results, 'melBands')

    run(loader)

    return results['melBands']
