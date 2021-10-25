#/usr/bin/env python

import torch
import torchaudio
import numpy as np

from ptpt.trainer import Trainer, TrainerConfig
from ptpt.callbacks import CallbackType
from ptpt.log import debug

from vqcpc import WaveVQCPC
from data import WaveDataset
from utils import set_seed

import argparse
import toml
from pathlib import Path
from types import SimpleNamespace

def main(args):
    torchaudio.set_audio_backend('soundfile')
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    cfg = SimpleNamespace(**toml.load(args.cfg_path))

    data_kwargs = {
        'nb_samples': cfg.data['nb_samples'],
        'sample_rate': cfg.data['sample_rate'],
        'shuffle_prob': cfg.data['shuffle_prob'],
        'polarity_prob': cfg.data['polarity_prob'],
        'noise_prob': cfg.data['noise_prob'],
        'gain_prob': cfg.data['gain_prob'],
    }

    train_dataset = WaveDataset(Path(cfg.data['root']) / 'training_set', **data_kwargs)
    test_dataset = WaveDataset(Path(cfg.data['root']) / 'development_set', **data_kwargs)

    def loss_fn(net, x):
        _, _, loss, nce_loss, l_loss, nce_acc = net(x)
        return loss, nce_loss, l_loss, nce_acc*100.

    net = WaveVQCPC(**cfg.vqcpc)

    trainer_cfg = TrainerConfig(
        **cfg.trainer,
        nb_workers = args.nb_workers,
        save_outputs = not args.no_save,
        use_cuda = not args.no_cuda,
        use_amp = not args.no_amp,
    )

    trainer = Trainer(
        net = net,
        loss_fn = loss_fn,
        train_dataset = train_dataset,
        test_dataset = test_dataset,
        cfg = trainer_cfg,
    )

    if args.resume:
        trainer.load_checkpoint(args.resume)

    if cfg.vqcpc['quantize_codes'] and cfg.vqcpc['quantize_mode'] == 'gumbel':
        def gumbel_callback(trainer):
            ts = trainer.nb_updates
            trainer.net.codebook.update_tau(ts)
            debug(f"decaying tau in gumbel quantizer. tau: {trainer.net.codebook.tau}")

        trainer.register_callback(
            CallbackType.ParameterUpdate, 
            gumbel_callback, 
            frequency=100
        )

    trainer.train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg-path', type=str, default='config/vqcpc/debug.toml')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--no-save', action='store_true')
    parser.add_argument('--no-cuda', action='store_true')
    parser.add_argument('--no-amp', action='store_true')
    parser.add_argument('--nb-workers', type=int, default=8)
    parser.add_argument('--detect-anomaly', action='store_true') # menacing aura!
    parser.add_argument('--seed', type=int, default=12345)
    args = parser.parse_args()

    set_seed(args.seed)

    main(args)
