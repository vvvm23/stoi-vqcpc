import torch
import torch.nn as nn

from functools import partial
from typing import List

from .utils import HelperModule

class Encoder(HelperModule):
    def build(self,
        in_channels:            int = 2,
        hidden_channels:        int = 512,
        kernel_strides:         List[int] = [5, 4, 2, 2, 2],
        kernel_sizes:           List[int] = [10, 8, 4, 4, 4],
        embedding_dim:          int = 64,
        norm_mode:              str = 'batch',
        dropout:                float = 0.1,
    ):
        nb_layers = len(kernel_sizes)
        assert len(kernel_strides) == len(kernel_sizes), "mismatch between number of provided kernel strides and sizes."
        assert norm_mode in ('batch', 'group', 'none', None), "invalid encoder normalization mode"

        if isinstance(hidden_channels, int):
            hidden_channels = [hidden_channels]*nb_layers
        hidden_channels = [in_channels] + hidden_channels

        norm_builder = nn.Identity
        if norm_mode == 'batch':
            norm_builder = partial(nn.BatchNorm1d, eps=1e-4)
        elif norm_mode == 'group':
            norm_builder = partial(nn.GroupNorm, num_groups=1, eps=1e-4)
        size_kwarg = 'num_features' if norm_mode == 'batch' else 'num_channels'

        conv = []
        for i in range(0, nb_layers):
            conv_args = (
                hidden_channels[i], 
                hidden_channels[i+1],
                kernel_sizes[i],
                kernel_strides[i],
                max(kernel_sizes[i] // 2 - 1, 0), # padding
            )

            l = nn.Sequential(
                nn.Conv1d(*conv_args),
                nn.Dropout(dropout),
                norm_builder(**{size_kwarg: hidden_channels[i+1]}),
                nn.ReLU()
            )
            torch.nn.init.kaiming_uniform_(l[0].weight.data, nonlinearity='relu')
            conv.append(l)
        self.conv = nn.Sequential(*conv)

        self.embedding_proj = nn.Linear(hidden_channels[-1], embedding_dim)

    def forward(self, x):
        x = self.conv(x)
        return self.embedding_proj(x.transpose(1, 2))

