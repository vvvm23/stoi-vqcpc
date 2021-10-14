import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from typing import List, Tuple
from functools import lru_cache, partial

from .utils import HelperModule
from .encoder import Encoder
from .vqvae import EMAQuantizer, GumbelQuantizer
from .aggregator import AggregatorGPT, AggregatorGRU
from .nce import InfoNCE

class WaveVQCPC(HelperModule):
    """
    `nn.Module` implementing VQ-CPC on raw audio waveforms.
    It is a mixture of a few implementations. They will be cited later.
    """
    def build(self,
            in_channels:                int = 2,

            encoder_channels:           int = 512,
            encoder_kernel_strides:     List[int] = [5, 4, 2, 2, 2],
            encoder_kernel_sizes:       List[int] = [10, 8, 4, 4, 4],
            encoder_norm_mode:          str = 'batch',

            quantize_codes:             bool = True,
            quantize_mode:              str = 'kmeans',
            gumbel_temp:                Tuple[float] = (1.0, 1.0, 1.0),
            nb_code_entries:            int = 512,
            embedding_dim:              int = 64,

            aggregator_dim:             int = 256,
            aggregator_layers:          int = 2,
            aggregator_mode:            str = 'attn',
            aggregator_mlp_dim:         int = 256,
            aggregator_seq_len:         int = 128,
            
            nce_steps:                  int = 12,
            nb_negatives:               int = 10,

            dropout:                    float = 0.1,
        ):
        """
        `WaveVQCPC` initialisation function

        Args:
            in_channels: number of channels in the input waveform

            encoder_channels: number of channels in convolutional encoder
            encoder_kernel_strides: stride size of each layer in the encoder
            encoder_kernel_sizes: kernel size of each layer in the encoder
            nb_code_entries: number of entries in VQ codebook
            embedding_dim: size of feature vectors in codebook

            aggregator_dim: dimension of aggregator output (GRU channels, attention dim, etc.)
            aggregator_mode: switch to GPT/conv/GRU aggregator (not implemented)

            nce_steps: number of InfoNCE steps in the future to consider
            nb_negatives: number of negative samples to get per step

            dropout: dropout rate
        """
        
        ALLOWED_AGGREGATOR = ['gru', 'attn']
        ALLOWED_QUANTIZER = ['kmeans', 'gumbel']
        assert aggregator_mode in ALLOWED_AGGREGATOR, f"invalid aggregator mode. expected one of: '{ALLOWED_AGGREGATOR}'"
        assert quantize_mode in ALLOWED_QUANTIZER, f"invalid quantization mode. expected one of: '{ALLOWED_QUANTIZER}'"

        self.encoder = Encoder(
            in_channels = in_channels,
            hidden_channels = encoder_channels,
            kernel_strides = encoder_kernel_strides,
            kernel_sizes = encoder_kernel_sizes,
            embedding_dim = embedding_dim,
            norm_mode = encoder_norm_mode,
            dropout = dropout,
        )

        self.codebook = None
        if quantize_codes:
            if quantize_mode == 'kmeans':
                self.codebook = EMAQuantizer(
                    nb_entries = nb_code_entries,
                    embedding_dim = embedding_dim,
                )
            elif quantize_mode == 'gumbel':
                self.codebook = GumbelQuantizer(
                    nb_entries = nb_code_entries,
                    tau = gumbel_temp,
                    embedding_dim = embedding_dim,
                )

        if aggregator_mode == 'gru':
            self.aggregator = AggregatorGRU(
                in_channels = embedding_dim,
                hidden_channels = aggregator_dim,
                nb_layers = aggregator_layers,
            )
        elif aggregator_mode == 'attn':
            self.aggregator = AggregatorGPT(
                embed_dim = embedding_dim,
                out_dim = aggregator_dim,
                mlp_dim = aggregator_mlp_dim,
                nb_layers = aggregator_layers,
                seq_len = aggregator_seq_len,
                dropout = dropout,
            )

        self.infonce = InfoNCE(
            c_dim = aggregator_dim, 
            z_dim = embedding_dim,
            k_steps = nce_steps,
            nb_negatives = nb_negatives,
        )

    def forward(self, x):
        """
        training forward pass of `WaveVQCPC`

        Args:
            x: raw audio waveform of shape (batch_size, in_channels, time_steps)

        Returns:
            z: embedding vectors of shape (batch_size, time_steps / downscale_rate, embedding_dim)
            c: context vectors of shape (batch_size, time_steps /  downscale_rate, aggregator_channels)
            loss: combined nce and latent loss
            nce_loss: nce loss
            l_loss: latent loss from VQ codebook, discounted by beta
            perplexity: codebook perplexity
        """
        z = self.encoder(x)

        l_loss, perplexity = 0.0, 0.0
        if self.codebook:
            z, l_loss, _ = self.codebook(z)

        c = self.aggregator(z)
        nce_loss, nce_acc = self.infonce(z, c)

        return z, c, nce_loss + l_loss, nce_loss, l_loss, nce_acc

    def encode(self, x):
        z = self.encoder(x)
        if self.codebook:
            z, _, _ = self.codebook(z)
        c = self.aggregator(z)
        return z, c
