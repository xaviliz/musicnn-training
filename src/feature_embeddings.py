from pathlib import Path
import json

from essentia.standard import TensorflowPredict, TensorTranspose
from essentia import Pool
import numpy as np

from feature_melspectrogram import (
    MelSpectrogramVGGish,
    MelSpectrogramMusiCNN,
    MelSpectrogramOpenL3,
)


class EmbeddingFromMelSpectrogram:
    def __init__(self, model_type, hop_time=1, batch_size=60, models_path='models/'):
        self.model_type = model_type
        self.hop_time = hop_time
        self.batch_size = batch_size
        self.models_path = models_path

        with open(Path(self.models_path, "models_config.json"), "r") as config_file:
            config = json.load(config_file)
        self.config = config[self.model_type]

        self.graph_filename = self.config["filename"]
        self.graph_path = Path(self.models_path, self.graph_filename)

        self.x_size = self.config["x_size"]
        self.y_size = self.config["y_size"]

        self.input_layer = self.config["input"]
        self.output_layer = self.config["embeddings"]

        self.seconds_to_patches = self.config["seconds_to_patches"]

        if self.model_type in ("musicnn", "effnet_b0", "effnet_b0_3M"):
            self.mel_extractor = MelSpectrogramMusiCNN()
        elif self.model_type in ("vggish", "yamnet"):
            self.mel_extractor = MelSpectrogramVGGish()
        elif self.model_type == "openl3":
            self.mel_extractor = MelSpectrogramOpenL3(hop_time=self.hop_time)

        params = {
            "inputs": [self.input_layer],
            "outputs": [self.output_layer],
            "squeeze": self.config["squeeze"],
            "graphFilename": str(self.graph_path),
        }

        self.model = TensorflowPredict(**params)

        # For now we don't know how to convert EffNet from Pytorch to TensorFlow with
        # dynamic (i.e., arbitrary) batch size, so we use a fixed one.
        if self.model_type == "effnet_b0_3M":
            self.batch_size = 64

    def compute(self, audio_file):
        mel_spectrogram = self.mel_extractor.compute(audio_file)
        # in OpenL3 the hop size is computed in the feature extraction level
        if self.model_type == "openl3":
            hop_size_samples = self.x_size
        else:
            hop_size_samples = int(self.hop_time * self.seconds_to_patches)

        batch = self.__melspectrogram_to_batch(mel_spectrogram, hop_size_samples)

        pool = Pool()
        embeddings = []
        nbatches = int(np.ceil(batch.shape[0] / self.batch_size))
        for i in range(nbatches):
            start = i * self.batch_size
            end = min(batch.shape[0], (i + 1) * self.batch_size)
            pool.set(self.input_layer, batch[start:end])
            out_pool = self.model(pool)
            embeddings.append(out_pool[self.output_layer].squeeze())

        return np.vstack(embeddings)

    def __melspectrogram_to_batch(self, melspectrogram, hop_time):
        npatches = int(np.ceil((melspectrogram.shape[0] - self.x_size) / hop_time) + 1)
        batch = np.zeros([npatches, self.x_size, self.y_size], dtype="float32")
        for i in range(npatches):
            last_frame = min(i * hop_time + self.x_size, melspectrogram.shape[0])
            first_frame = i * hop_time
            data_size = last_frame - first_frame

            # the last patch may be empty, remove it and exit the loop
            if data_size <= 0:
                batch = np.delete(batch, i, axis=0)
                break
            else:
                batch[i, :data_size] = melspectrogram[first_frame:last_frame]

        batch = np.expand_dims(batch, 1)
        if self.config["permutation"]:
            batch = TensorTranspose(permutation=self.config["permutation"])(batch)
        return batch


class EmbeddingFromWaveForm:
    def __init__(self, model_type, config):
        self.model_type = model_type
        self.config = config
        raise NotImplementedError()
