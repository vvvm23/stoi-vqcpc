import torch
import torch.nn as nn

from .utils import HelperModule

# standard aggregator as in van den Oord, 2020
class AggregatorGRU(HelperModule):
    def build(self,
            in_channels:        int = 64,
            hidden_channels:    int = 256,
            nb_layers:          int = 2,
        ):
        self.gru = nn.GRU(in_channels, hidden_channels, batch_first=True, num_layers=nb_layers)

    def forward(self, z):
        return self.gru(z)[0]
