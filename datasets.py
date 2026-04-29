import glob
import itertools

import numpy as np
import scipy.signal
import soundfile as sf
import soxr
import torch

import utils


class WavFiles(torch.utils.data.Dataset):
    def __init__(
        self,
        sr=20000,
        filenames="data/*wav",
        dbspl=60,
        preload=True,
    ):
        """
        PyTorch Dataset object for loading wav files.
        """
        self.sr = sr
        self.filenames = sorted(glob.glob(filenames))
        self.preload = preload
        if self.preload:
            self.signals = []
            for filename in self.filenames:
                signal, sr_src = sf.read(filename)
                self.signals.append(soxr.resample(signal, sr_src, self.sr))
        self.rng = np.random.default_rng()
        if isinstance(dbspl, (tuple, list)):
            self.sample_dbspl = lambda: float(self.rng.uniform(*dbspl))
        else:
            self.sample_dbspl = lambda: dbspl

    def __getitem__(self, idx):
        """ """
        filename = self.filenames[idx]
        if self.preload:
            signal = self.signals[idx]
        else:
            signal, sr_src = sf.read(filename)
            signal = soxr.resample(signal, sr_src, self.sr)
        dbspl = self.sample_dbspl()
        out = {
            "sr": self.sr,
            "signal": signal.astype(np.float32),
            "dbspl": dbspl,
            "filename": filename,
        }
        return out

    def __len__(self):
        """ """
        return len(self.filenames)


class SyntheticVowels(torch.utils.data.Dataset):
    def __init__(
        self,
        sr=20000,
        dur=0.250,
        dur_ramp=0.025,
        dbspl=60,
        f0=100,
        n_examples=None,
    ):
        """
        Simple synthetic vowel generator using a source-filter model.
        Formant frequencies from Kewley-Port & Watson (1994, JASA).
        """
        self.sr = sr
        self.dur = dur
        self.dur_ramp = dur_ramp
        self.n_examples = n_examples
        if isinstance(dbspl, tuple):
            dbspl_min, dbspl_max = dbspl
            dbspl_list = None
        elif isinstance(dbspl, list):
            dbspl_min = dbspl_max = None
            dbspl_list = dbspl
        else:
            dbspl_min = dbspl_max = None
            dbspl_list = [dbspl]
        if isinstance(f0, tuple):
            f0_min, f0_max = f0
            f0_list = None
        elif isinstance(f0, list):
            f0_min = f0_max = None
            f0_list = f0
        else:
            f0_min = f0_max = None
            f0_list = [f0]
        vowel_list = list(range(10))
        # Evaluation mode = full grid of dbspl, f0, and vowel values
        self.eval_mode = (dbspl_list is not None) and (f0_list is not None)
        if self.eval_mode:
            grid = list(itertools.product(dbspl_list, f0_list, vowel_list))
            self.grid = np.array(grid, float)
        # Sampling functions for non-evaluation mode
        self.rng = np.random.default_rng()
        self.sample_vowel = lambda: int(self.rng.choice(vowel_list))
        if dbspl_list is not None:
            self.sample_dbspl = lambda: float(self.rng.choice(dbspl_list))
        else:
            self.sample_dbspl = lambda: float(self.rng.uniform(dbspl_min, dbspl_max))
        if f0_list is not None:
            self.sample_f0 = lambda: float(self.rng.choice(f0_list))
        else:
            self.sample_f0 = lambda: float(
                np.exp(self.rng.uniform(np.log(f0_min), np.log(f0_max)))
            )
        # Constants to be used for stimulus generation
        self.map_vowel_to_str = {
            0: "/i/ (heed)",
            1: "/I/ (hid)",
            2: "/e/ (hayed)",
            3: "/eps/ (head)",
            4: "/ae/ (had)",
            5: "/a/ (hod)",
            6: "/^/ (hud)",
            7: "/o/ (hoed)",
            8: "/U/ (hood)",
            9: "/u/ (who'd)",
        }
        self.FORMANT_FREQUENCIES = np.array(
            [
                (325, 2900, 3500),  # 0: /i/
                (450, 2300, 3000),  # 1: /I/
                (550, 2500, 3100),  # 2: /e/
                (600, 2200, 3000),  # 3: /eps/
                (1000, 1950, 3000),  # 4: /ae/
                (875, 1175, 2850),  # 5: /a/
                (700, 1400, 2600),  # 6: /^/
                (500, 800, 2000),  # 7: /o/
                (500, 1350, 2500),  # 8: /U/
                (250, 850, 2650),  # 9: /u/
            ]
        )
        self.FORMANT_BANDWIDTHS = np.array((70, 90, 170))
        self.t = np.arange(0, self.dur, 1 / self.sr)
        self.R = np.exp(-np.pi * self.FORMANT_BANDWIDTHS / self.sr)
        rise, fall = np.split(np.hanning(2 * int(self.dur_ramp * self.sr)), 2)
        self.ramp = np.ones_like(self.t)
        self.ramp[: len(rise)] *= rise
        self.ramp[-len(fall) :] *= fall
        b, a = scipy.signal.butter(
            N=8,
            Wn=4000,
            btype="low",
            fs=self.sr,
        )
        self.lowpass_filter = lambda x: scipy.signal.lfilter(b, a, x)

    def generate_signal(self, vowel, f0, dbspl):
        """
        Source-filter model: a harmonic glottal pulse
        train filtered with appropriate formant filter.
        """
        F = self.FORMANT_FREQUENCIES[vowel]
        theta = 2 * np.pi * F / self.sr
        poles = self.R * np.exp(1j * theta)
        poles = np.concatenate([poles, np.conj(poles)])
        b, a = scipy.signal.zpk2tf(0, poles, 1)
        x = np.zeros_like(self.t)
        period = int(np.round(self.sr / f0))
        x[np.random.randint(period) :: period] = 1
        x = scipy.signal.lfilter(b, a, x)
        x = self.lowpass_filter(x)
        x = utils.set_dbspl(x * self.ramp, dbspl)
        return x

    def __getitem__(self, idx):
        """ """
        if self.eval_mode:
            dbspl, f0, vowel = self.grid[idx]
            vowel = int(vowel)
        else:
            dbspl = self.sample_dbspl()
            f0 = self.sample_f0()
            vowel = int(self.sample_vowel())
        out = {
            "sr": self.sr,
            "signal": self.generate_signal(vowel, f0, dbspl).astype(np.float32),
            "dbspl": dbspl,
            "f0": f0,
            "formants": self.FORMANT_FREQUENCIES[vowel],
            "vowel": vowel,
            "vowel_str": self.map_vowel_to_str[vowel],
        }
        return out

    def __len__(self):
        """ """
        return len(self.grid) if self.eval_mode else self.n_examples
