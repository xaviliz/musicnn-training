
# Work derivated from https://github.com/p-lambda/jukemir/

from essentia.standard import MonoLoader
import numpy as np
import torch

from jukebox.hparams import Hyperparams, setup_hparams
from jukebox.make_models import MODELS, make_prior, make_vqvae
from jukebox.utils.dist_utils import setup_dist_from_mpi

JUKEBOX_SAMPLE_RATE = 44100
T = 8192


class Jukebox():
    def __init__(self):
        # Set up MPI
        self.rank, self.local_rank, self.device = setup_dist_from_mpi()

        # Set up VQVAE
        self.model = "5b"  # or "1b_lyrics"
        self.hps = Hyperparams()
        self.hps.sr = 44100
        self.hps.n_samples = 3 if self.model == "5b_lyrics" else 8
        self.hps.name = "samples"
        self.chunk_size = 16 if self.model == "5b_lyrics" else 32
        self.max_batch_size = 3 if self.model == "5b_lyrics" else 16
        self.hps.levels = 3
        self.hps.hop_fraction = [0.5, 0.5, 0.125]
        vqvae, *priors = MODELS[self.model]
        self.vqvae = make_vqvae(
            setup_hparams(vqvae, dict(sample_length=1048576)), self.device
        )

        # Set up language model
        self.hparams = setup_hparams(priors[-1], dict())
        self.hparams["prior_depth"] = 36
        self.top_prior = make_prior(self.hparams, self.vqvae, self.device)

        # get conditioning info
        self.x_cond, self.y_cond = self.get_cond()

    @staticmethod
    def load_audio_from_file(fpath, constrain_size=True):
        audio = MonoLoader(filename=fpath, sampleRate=JUKEBOX_SAMPLE_RATE)()
        # audio, _ = lr.load(fpath, sr=JUKEBOX_SAMPLE_RATE)
        if audio.ndim == 1:
            audio = audio[np.newaxis]
        audio = audio.mean(axis=0)

        # normalize audio
        norm_factor = np.abs(audio).max()
        if norm_factor > 0:
            audio /= norm_factor

        return audio.flatten()

    def get_z(self, audio):
        # don't compute unnecessary discrete encodings
        audio = audio[: JUKEBOX_SAMPLE_RATE * 25]

        zs = self.vqvae.encode(torch.cuda.FloatTensor(audio[np.newaxis, :, np.newaxis]))

        z = zs[-1].flatten()[np.newaxis, :]

        if z.shape[-1] < 8192:
            raise ValueError("Audio file is not long enough")

        return z

    def get_cond(self):
        sample_length_in_seconds = 62

        self.hps.sample_length = (
            int(sample_length_in_seconds * self.hps.sr) // self.top_prior.raw_to_tokens
        ) * self.top_prior.raw_to_tokens

        # NOTE: the 'lyrics' parameter is required, which is why it is included,
        # but it doesn't actually change anything about the `x_cond`, `y_cond`,
        # nor the `prime` variables
        metas = [
            dict(
                artist="unknown",
                genre="unknown",
                total_length=self.hps.sample_length,
                offset=0,
                lyrics="""lyrics go here!!!""",
            ),
        ] * self.hps.n_samples

        labels = [None, None, self.top_prior.labeller.get_batch_labels(metas, "cuda")]

        x_cond, y_cond, prime = self.top_prior.get_cond(None, self.top_prior.get_y(labels[-1], 0))

        x_cond = x_cond[0, :T][np.newaxis, ...]
        y_cond = y_cond[0][np.newaxis, ...]

        return x_cond, y_cond

    def get_final_activations(self, z, x_cond, y_cond):
        x = z[:, :T]

        # make sure that we get the activations
        self.top_prior.prior.only_encode = True

        out = self.top_prior.prior.forward(
            x, x_cond=x_cond, y_cond=y_cond, encoder_kv=None, fp16=False
        )

        return out

    def compute_and_save(self, input_path, output_path, meanpool=True, constrain_size=True):
        """Decode, resample, convert to mono, and normalize audio"""
        representation = self.compute(
            input_path,
            meanpool=meanpool,
            constrain_size=constrain_size,
        )

        np.save(output_path, representation)

    def compute(self, fpath, meanpool=True, constrain_size=True):
        audio = self.load_audio_from_file(fpath)

        with torch.no_grad():
            return self.get_acts_from_audio(
                audio,
                meanpool=meanpool,
                constrain_size=constrain_size
            )

    def get_acts_from_audio(self, audio, meanpool=True, constrain_size=True):
        # constrain to 24s in the middle
        if constrain_size:
            tgt_duration = JUKEBOX_SAMPLE_RATE * 24
            if len(audio) < tgt_duration:
                min_duration = 3  # arbitrary, set to low to be able to process most of tonal_atonal
                assert len(audio) / JUKEBOX_SAMPLE_RATE > min_duration, f"audio length is less than {min_duration} seconds!"
                audio = np.tile(audio, int(np.ceil(tgt_duration / len(audio))))[:tgt_duration]  # so bad that numpy.pad doesn't have a repeat mode!
            else:
                middle = len(audio) // 2
                audio = audio[middle - tgt_duration // 2: middle + tgt_duration // 2 + 1]

        # run vq-vae on the audio
        z = self.get_z(audio)

        # get the activations from the LM
        acts = self.get_final_activations(z, self.x_cond, self.y_cond)

        # postprocessing
        acts = acts.squeeze().type(torch.float32)

        if meanpool:
            acts = acts.mean(dim=0)

        acts = np.array(acts.cpu())

        return acts
