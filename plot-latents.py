#/usr/bin/env python

import torch

from vqcpc import WaveVQCPC
from stoi import STOIPredictor
from data import WaveDataset
from utils import get_device

import matplotlib.pyplot as plt
import argparse
import toml
import random
from pathlib import Path
from types import SimpleNamespace

def print_prompt():
    msg = (
        "1) View latents\n"
        "9) Next sample\n"
        "0) Exit\n"
    )
    print(msg)

# https://pytorch.org/tutorials/beginner/audio_preprocessing_tutorial.html
def plot_waveform(waveform, sample_rate, xlim=None, ylim=None, axes=None):
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sample_rate

    if axes is None:
        figure, axes = plt.subplots(1, 1)
    for c in range(num_channels):
        axes.plot(time_axis, waveform[c], linewidth=1, color='blue' if c else 'red', alpha=0.5)
        axes.set_xlim(left=0, right=num_frames/sample_rate)
        axes.grid(True)

@torch.no_grad()
def get_embeddings(net, wav):
    z = net.encoder(wav).squeeze(0)
    c = net.aggregator(z.unsqueeze(0)).squeeze(0)
    return z, c

@torch.no_grad()
def get_score(net, c):
    frame_scores = net(c)
    return frame_scores

def main(args):
    plt.style.use(args.style)
    cfg = SimpleNamespace(**toml.load(args.cfg_path))
    device = get_device(args.no_cuda)

    data_kwargs = {
        'nb_samples': cfg.data['nb_samples'],
        'sample_rate': cfg.data['sample_rate'],
        'shuffle_prob': 0.0,
        'polarity_prob': 0.0,
        'noise_prob': 0.0,
        'gain_prob': 0.0,
        'return_file': True,
    }
    dataset = WaveDataset(args.data_root, **data_kwargs)
    dataset_length = len(dataset)

    chk = torch.load(args.checkpoint, map_location=device)
    net = WaveVQCPC(**cfg.vqcpc).to(device)
    net.load_state_dict(chk['net'])
    net.eval()

    stoi = None
    if args.stoi:
        stoi_cfg = SimpleNamespace(**toml.load(args.stoi_cfg))
        chk = torch.load(args.stoi, map_location=device)
        stoi = STOIPredictor(**stoi_cfg.model).to(device)
        stoi.load_state_dict(chk['net'])
        stoi.eval()

    action = 999
    while not action == 0:
        x, name = dataset.__getitem__(random.randint(0, dataset_length), crop=False)
        x = x.unsqueeze(0)

        z, c = get_embeddings(net, x.to(device))
        
        print(f"> name: {name}")
        print(f"> z vector sequence {z.shape}")
        print(z)
        print()
        print(f"> c vector sequence {c.shape}")
        print(c)
        if stoi:
            frame_scores = get_score(stoi, c)
            print(f"> predicted intelligibility score: {frame_scores.mean()}")
        while True:
            try:
                print_prompt()
                action = int(input("> "))
            except:
                print("! invalid prompt input")
                continue

            if action == 1:
                if stoi and not cfg.model['pool']:
                    fig, axes = plt.subplots(3, 1)
                    plot_waveform(x.squeeze().cpu(), 16_000, axes=axes[0])
                    axes[0].set_xlabel("time (s)")
                    axes[0].set_ylabel("amplitude (s)")

                    axes[1].plot(frame_scores.squeeze().cpu())
                    axes[1].set_xlim(left=0, right=frame_scores.shape[-1])
                    axes[1].set_xlabel("frames")
                    axes[1].set_ylabel("intelligibility score")

                    axes[2].imshow(c.squeeze().transpose(0, 1).cpu(), cmap=args.cmap, aspect='auto')
                    axes[2].set_xlabel("frames")
                    axes[2].set_yticks([])
                else:
                    fig, axes = plt.subplots(3, 1)
                    plot_waveform(x.squeeze().cpu(), 16_000, axes=axes[0])
                    axes[0].set_xlabel("time")
                    axes[0].set_ylabel("amplitude (s)")

                    axes[1].imshow(z.squeeze().transpose(0, 1).cpu(), cmap=args.cmap, aspect='auto')
                    axes[1].set_xlabel("frames")
                    axes[2].set_yticks([])

                    axes[2].imshow(c.squeeze().transpose(0, 1).cpu(), cmap=args.cmap, aspect='auto')
                    axes[2].set_xlabel("frames")
                    axes[2].set_yticks([])
                
                fig.tight_layout()
                plt.show()

            if action in [0, 9]:
                break

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', type=str)
    parser.add_argument('data_root', type=str)
    parser.add_argument('--cfg-path', type=str, default='config/vqcpc/vqcpc-gru128-kmean.toml')
    parser.add_argument('--no-cuda', action='store_true')
    parser.add_argument('--no-amp', action='store_true')
    parser.add_argument('--cmap', type=str, default='viridis')
    parser.add_argument('--style', type=str, default='dark_background')
    parser.add_argument('--stoi', type=str, default=None) # path to some STOI score predictor
    parser.add_argument('--stoi-cfg', type=str, default=None)
    args = parser.parse_args()
    main(args)
