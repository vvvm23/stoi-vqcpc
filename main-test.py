#/usr/bin/env python
import torch
import torchaudio
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

import numpy as np
from scipy.stats import spearmanr

from ptpt.log import info

import tqdm
import argparse
import toml
import datetime
from pathlib import Path
from types import SimpleNamespace

from data import ConcatDataset, FirstChannelDataset, FeatureScoreDataset
from stoi import STOIPredictor
from utils import set_seed, get_device

def main(args):
    torchaudio.set_audio_backend('soundfile')
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    cfg = SimpleNamespace(**toml.load(args.cfg_path))

    device = get_device(args.no_cuda)

    args.data_root = Path(args.data_root)
    data_mode = cfg.data['mode']
    if data_mode in ['vqcpc']:
        test_dataset = FeatureScoreDataset(args.data_root, load_z=False, return_file=True)
    elif data_mode in ['concat']:
        test_dataset = ConcatDataset(args.data_root, cfg.data['sample_rate'], return_file=True)
    elif data_mode in ['single', 'rossbach']:
        test_dataset = FirstChannelDataset(args.data_root, cfg.data['sample_rate'], return_file=True)

    def loss_fn(net, batch):
        x, stoi, end = batch
        batch_size = x.shape[0]
        stoi_pred = net(x)
        
        if not cfg.model['pool']:
            stoi_pred = masked_mean(stoi_pred, end)
        loss = F.mse_loss(stoi_pred, stoi)

        stoi, stoi_pred = stoi.detach().cpu(), stoi_pred.detach().cpu()
        return loss, np.corrcoef(stoi, stoi_pred)[0, 1], spearmanr(stoi.T, stoi_pred.T)[0], stoi_pred

    def collate_fn(data):
        x, stoi, name = zip(*data)

        lengths = [v.shape[0] for v in x]
        batch_size = len(lengths)

        stoi = torch.FloatTensor(stoi)
        lengths = torch.LongTensor(lengths)

        X = pad_sequence(x, batch_first=True)

        return X, stoi, lengths, name

    test_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        num_workers=args.nb_workers, 
        pin_memory=True, 
        collate_fn=collate_fn
    )

    chk = torch.load(args.checkpoint_path)
    net = STOIPredictor(**cfg.model).to(device)
    net.load_state_dict(chk['net'])
    net.eval()

    if not args.no_save:
        ctime = str(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        result_f = open(f"results_{cfg.trainer['exp_name']}_{args.data_root.name}_{ctime}.csv", mode='w')

    pb = tqdm.tqdm(test_loader)
    total_loss, total_lcc, total_srcc = 0.0, 0.0, 0.0
    with torch.inference_mode(), torch.cuda.amp.autocast(enabled = not args.no_amp and not args.no_cuda):
        for batch in pb:
            name = batch[-1]
            loss, lcc, srcc, score = loss_fn(net, batch[:-1])
            total_loss += loss.item()
            total_lcc += lcc
            total_srcc += srcc

            if not args.no_save:
                for n, s in zip(name, score):
                    row = (
                        f"{n},"
                        f"{s.item()}\n"
                    )
                    result_f.write(row)

    if not args.no_save:
        result_f.close()

    avg_loss = total_loss / len(test_loader)
    avg_lcc = total_lcc / len(test_loader)
    avg_srcc = total_srcc / len(test_loader)

    info(f"average MSE loss: {avg_loss}")
    info(f"average LCC: {avg_lcc}")
    info(f"average SRCC: {avg_srcc}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint_path', type=str)
    parser.add_argument('data_root', type=str)
    parser.add_argument('--cfg-path', type=str, default='config/vqcpc/stoi-gru128-pool-kmean.toml')
    parser.add_argument('--no-save', action='store_true')
    parser.add_argument('--no-cuda', action='store_true')
    parser.add_argument('--no-amp', action='store_true')
    parser.add_argument('--no-tqdm', action='store_true')
    parser.add_argument('--nb-workers', type=int, default=8)
    parser.add_argument('--detect-anomaly', action='store_true') # menacing aura!
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--seed', type=int, default=12345)
    args = parser.parse_args()

    set_seed(args.seed)

    main(args)
