#/usr/bin/env python

import torch
import torchaudio

import numpy as np

import csv
import toml
import argparse
from pathlib import Path
from tqdm import tqdm
from types import SimpleNamespace

from vqcpc import WaveVQCPC
from utils import get_device

def main(args):
    torchaudio.set_audio_backend('soundfile')
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    cfg = SimpleNamespace(**toml.load(args.cfg_path))
    device = get_device(args.no_cuda)

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)

    net = WaveVQCPC(**cfg.vqcpc).to(device)
    chk = torch.load(args.model_path)
    net.load_state_dict(chk['net'])
    net.eval()

    meta_f = open(data_dir / 'meta.csv', mode='r')
    csv_reader = csv.reader(meta_f)

    if args.array_rank is not None:
        out_f = open(out_dir / f'meta-{args.array_rank}.csv', mode='w')
        csv_contents = list(csv_reader)[args.array_rank::args.array_size]
    else:
        out_f = open(out_dir / 'meta.csv', mode='w')
        csv_contents = list(csv_reader)

    pb = tqdm(enumerate(csv_contents), total=len(csv_contents), disable=args.no_tqdm)
    for i, (name, *_, stoi) in pb:
        pb.set_description(name)
        # with torch.no_grad(), torch.cuda.amp.autocast(enabled = not args.no_amp and not args.no_cuda):
        with torch.inference_mode(), torch.cuda.amp.autocast(enabled = not args.no_amp and not args.no_cuda):
            wav, rate = torchaudio.load(data_dir / name)
            wav = wav.to(device)
            assert rate == cfg.data['sample_rate'], "sample rate did not match" 

            if wav.shape[-1] < cfg.data['nb_samples']:
                padded = torch.zeros(2, cfg.data['nb_samples']).to(device)
                padded[:, :wav.shape[-1]] = wav
                wav = padded

            z = net.encoder(wav.unsqueeze(0)).squeeze(0)
            c = net.aggregator(z.unsqueeze(0)).squeeze(0)
        
        file_name = Path(name).with_suffix('').name
        np.save((out_dir / file_name).with_suffix('.z.npy'), z.cpu().numpy())
        np.save((out_dir / file_name).with_suffix('.c.npy'), c.cpu().numpy())

        meta_string = (
            f"{(out_dir / file_name).with_suffix('.z.npy')},"
            f"{(out_dir / file_name).with_suffix('.c.npy')},"
            f"{stoi}\n"
        )
        out_f.write(meta_string)

        if args.no_tqdm:
            print(f"{i+1}/{len(csv_contents)} processed. wrote {z.shape[0]} frames.")

    out_f.close()
    meta_f.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', type=str)
    parser.add_argument('data_dir', type=str)
    parser.add_argument('out_dir', type=str)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--cfg-path', type=str, default='config/vqcpc/debug.toml')
    parser.add_argument('--no-cuda', action='store_true')
    parser.add_argument('--no-amp', action='store_true')
    parser.add_argument('--no-tqdm', action='store_true')
    parser.add_argument('--detect-anomaly', action='store_true') # menacing aura!
    parser.add_argument('--array-size', '-n', type=int, default=None)
    parser.add_argument('--array-rank', '-r', type=int, default=None)
    args = parser.parse_args()
    main(args)
