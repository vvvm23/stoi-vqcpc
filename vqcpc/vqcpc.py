import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List, Tuple

from .utils import HelperModule
from .encoder import Encoder
from .vqvae import EMAQuantizer
from .aggregator import AggregatorGRU
from .nce import InfoNCE

class WaveVQCPC(HelperModule):
    """
    `nn.Module` implementing VQ-CPC on raw audio waveforms.
    """
    def build(self,
            in_channels:                int = 2,

            encoder_channels:           int = 512,
            encoder_kernel_strides:     List[int] = [5, 4, 2, 2, 2],
            encoder_kernel_sizes:       List[int] = [10, 8, 4, 4, 4],
            encoder_norm_mode:          str = 'batch',

            quantize_codes:             bool = True,
            nb_code_entries:            int = 512,
            embedding_dim:              int = 64,

            aggregator_dim:             int = 256,
            aggregator_layers:          int = 2,
            
            nce_steps:                  int = 12,
            nb_negatives:               int = 10,

            dropout:                    float = 0.1,

            **kwargs
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

            nce_steps: number of InfoNCE steps in the future to consider
            nb_negatives: number of negative samples to get per step

            dropout: dropout rate
        """
        
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
            self.codebook = EMAQuantizer(
                nb_entries = nb_code_entries,
                embedding_dim = embedding_dim,
            )

        self.aggregator = AggregatorGRU(
            in_channels = embedding_dim,
            hidden_channels = aggregator_dim,
            nb_layers = aggregator_layers,
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
