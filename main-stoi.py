#/usr/bin/env python
import torch
import torchaudio
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

import numpy as np
from scipy.stats import spearmanr

from ptpt.trainer import Trainer, TrainerConfig

import argparse
import toml
from pathlib import Path
from types import SimpleNamespace

from data import ConcatDataset, FirstChannelDataset, FeatureScoreDataset
from stoi import STOIPredictor
from utils import set_seed

def masked_mean(x, end):
    N, D = x.shape
    idx = torch.arange(D).to(x.device).view(1, -1).repeat(N, 1)
    total = end.float()
    end = end.view(-1, 1).repeat(1, D)
    mask = (idx < end).float()

    return (x*mask).sum(dim=-1) / total

def main(args):
    torchaudio.set_audio_backend('soundfile')
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    cfg = SimpleNamespace(**toml.load(args.cfg_path))

    data_mode = cfg.data['mode']
    if data_mode in ['vqcpc']:
        train_dataset = FeatureScoreDataset(cfg.data['train_root'], load_z=False)
        test_dataset = FeatureScoreDataset(cfg.data['dev_root'], load_z=False)
    elif data_mode in ['concat']:
        train_dataset = ConcatDataset(cfg.data['train_root'], cfg.data['sample_rate'])
        test_dataset = ConcatDataset(cfg.data['dev_root'], cfg.data['sample_rate'])
    elif data_mode in ['single', 'rossbach']:
        train_dataset = FirstChannelDataset(cfg.data['train_root'], cfg.data['sample_rate'])
        test_dataset = FirstChannelDataset(cfg.data['dev_root'], cfg.data['sample_rate'])

    def loss_fn(net, batch):
        x, stoi, end = batch
        if args.no_amp:
            x = x.float()        

        batch_size = x.shape[0]
        stoi_pred = net(x)
        
        if not cfg.model['pool']:
            stoi_pred = masked_mean(stoi_pred, end)
        loss = F.mse_loss(stoi_pred, stoi)

        stoi, stoi_pred = stoi.detach().cpu(), stoi_pred.detach().cpu()
        return loss, np.corrcoef(stoi, stoi_pred)[0, 1], spearmanr(stoi.T, stoi_pred.T)[0]

    def collate_fn(data):
        x, stoi = zip(*data) 
        lengths = [v.shape[0] for v in x]
        batch_size = len(lengths)

        stoi = torch.FloatTensor(stoi)
        lengths = torch.LongTensor(lengths)

        X = pad_sequence(x, batch_first=True)
        return X, stoi, lengths

    net = STOIPredictor(**cfg.model)

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
        collate_fn = collate_fn,
        cfg = trainer_cfg,
    )

    if args.resume:
        trainer.load_checkpoint(args.resume)

    trainer.train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg-path', type=str, default='config/debug.toml')
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
