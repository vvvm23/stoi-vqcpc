import torch
import torch.nn.functional as F
import torchaudio
import numpy as np

import csv
import random
from pathlib import Path
from collections import namedtuple

from torch_audiomentations import Compose, AddColoredNoise, Gain, PolarityInversion, ShuffleChannels

class FeatureScoreDataset(torch.utils.data.Dataset):
    def __init__(self,
        root: str,
        load_z: bool = True,
        load_c: bool = True,
        return_file: str = False,
    ):
        super().__init__()

        if not (load_z or load_c):
            raise ValueError("must load one or both of z and c vectors")
        self.load_z = load_z
        self.load_c = load_c
        self.load_all = load_z and load_c
        self.return_file = return_file

        if isinstance(root, str):
            root = Path(root)
        self.root = root
        meta_file = root / 'meta.csv'

        Entry = namedtuple('Entry', ['z', 'c', 'stoi'])
        with open(meta_file, mode='r') as f:
            csv_reader = csv.reader(f)
            self.zc_score = [Entry(*row) for row in csv_reader]

    def __len__(self):
        return len(self.zc_score)

    def __getitem__(self, idx):
        entry = self.zc_score[idx]
        stoi = float(entry.stoi)
        
        if self.load_all:
            z = np.load(entry.z)
            c = np.load(entry.c)
            batch = [torch.from_numpy(z), torch.from_numpy(c), stoi]
        elif self.load_z:
            z = np.load(entry.z)
            batch = [torch.from_numpy(z), stoi]
        elif self.load_c:
            c = np.load(entry.c)
            batch = [torch.from_numpy(c), stoi]

        if self.return_file:
            name = Path(Path(entry.z).stem).stem
            batch.append(name)
        return batch

class WaveDataset(torch.utils.data.Dataset):
    def __init__(self,
        root: str,
        nb_samples: int = 20480,
        sample_rate: int = 16_000,
        shuffle_prob: float = 0.5,
        polarity_prob: float = 0.5,
        noise_prob: float = 0.5,
        gain_prob: float = 0.5,
        return_file: bool = False,
    ):
        super().__init__()
        if isinstance(root, str):
            root = Path(root)
        self.root = root
        self.sample_rate = sample_rate
        self.nb_samples = nb_samples
        self.return_file = return_file

        self.augmentations = Compose(
            transforms = [
                ShuffleChannels(p=shuffle_prob),
                PolarityInversion(p=polarity_prob),
                AddColoredNoise(p=noise_prob),
                Gain(p=gain_prob),
            ]
        )

        meta_file = self.root / 'meta.csv'
        f = open(meta_file, mode='r')
        csv_reader = csv.reader(f)
        self.wav_files = [name for name, *_ in csv_reader]

    def __len__(self):
        return len(self.wav_files)

    def __getitem__(self, idx, crop=True):
        wav, rate = torchaudio.load(self.root / self.wav_files[idx])
        assert rate == self.sample_rate, "sample rate did not match" 

        if crop:
            if wav.shape[-1] - self.nb_samples - 1 > 0:
                pos = random.randint(0, wav.shape[-1] - self.nb_samples - 1)
            else:
                pos = 0
            wav = wav[:, pos : pos + self.nb_samples]
            
            padded = torch.zeros(2, self.nb_samples)
            padded[:, :wav.shape[-1]] = wav
            wav = padded

        wav = self.augmentations(wav.unsqueeze(0), sample_rate=rate).squeeze()
        if self.return_file:
            return wav, self.root / self.wav_files[idx]
        return wav

class ConcatDataset(torch.utils.data.Dataset):
    def __init__(self,
            root: str,
            sample_rate: int = 16_000,
            return_file: bool = False,
        ):
        super().__init__()
        if isinstance(root, str):
            root = Path(root)
        self.root = root
        meta_file = root / 'meta.csv'

        self.sample_rate = sample_rate
        self.return_file = return_file

        self.augmentations = Compose(
            transforms = [
                ShuffleChannels(p=0.5),
                PolarityInversion(p=0.5),
            ]
        )

        self.melspec_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=512,
            n_mels=40,
            f_max=8000,
            win_length=320,
            hop_length=160,
        )
        self.delta_transform = torchaudio.functional.compute_deltas

        f = open(meta_file, mode='r')
        csv_reader = csv.reader(f)
        self.wav_score = [(n, s) for n, *_, s in csv_reader]
        f.close()

    def __len__(self):
        return len(self.wav_score)

    def __getitem__(self, idx):
        wav, rate = torchaudio.load(self.root / self.wav_score[idx][0])
        assert rate == self.sample_rate, "sample rate did not match" # TODO: do we need to have a rate attribute?
        wav = self.augmentations(wav.unsqueeze(0), sample_rate=rate).squeeze(0)
        f = self.melspec_transform(wav)
        fd = self.delta_transform(f)
        fdd = self.delta_transform(fd)

        feature = torch.cat([f,fd,fdd], dim=1).permute(2, 0, 1) # 240-dim feature
        if self.return_file:
            return feature, float(self.wav_score[idx][1]), self.root / self.wav_score[idx][0] 
        return feature, float(self.wav_score[idx][1])

class FirstChannelDataset(torch.utils.data.Dataset):
    def __init__(self,
            root: str,
            sample_rate: int = 16_000,
            return_file: bool = False,
        ):
        super().__init__()
        if isinstance(root, str):
            root = Path(root)
        self.root = root
        meta_file = root / 'meta.csv'

        self.sample_rate = sample_rate
        self.return_file = return_file

        self.augmentations = Compose(
            transforms = [
                PolarityInversion(p=0.5),
            ]
        )

        self.melspec_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=512,
            n_mels=40,
            f_max=8000,
            win_length=320,
            hop_length=160,
        )
        self.delta_transform = torchaudio.functional.compute_deltas

        f = open(meta_file, mode='r')
        csv_reader = csv.reader(f)
        self.wav_score = [(n, s) for n, *_, s in csv_reader]
        f.close()

    def __len__(self):
        return len(self.wav_score)

    def __getitem__(self, idx):
        wav, rate = torchaudio.load(self.root / self.wav_score[idx][0])
        assert rate == self.sample_rate, "sample rate did not match" # TODO: do we need to have a rate attribute?
        wav = wav[0] # only take first channel
        wav = self.augmentations(wav.view(1, 1, -1), sample_rate=rate).squeeze(0)
        f = self.melspec_transform(wav)
        fd = self.delta_transform(f)
        fdd = self.delta_transform(fd)

        feature = torch.cat([f,fd,fdd], dim=1).permute(2, 0, 1) # 120-dim feature
        if self.return_file:
            return feature, float(self.wav_score[idx][1]), self.root / self.wav_score[idx][0] 
        return feature, float(self.wav_score[idx][1])
