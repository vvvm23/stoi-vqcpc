import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import HelperModule

class STOIPredictor(HelperModule):
    def build(self,
            in_dim: int = 64,
            dropout: float = 0.3,
            norm: bool = True,
            sig_out: bool = False,
            small: bool = False,
            pool: bool = False,
            **kwargs,
        ):
        self.pool = None
        if pool:
            self.pool = nn.Linear(in_dim, 1)

        if small:
            self.layers = nn.Sequential(
                nn.LayerNorm(in_dim) if norm else nn.Identity(),
                nn.Linear(in_dim, 1),
                nn.Sigmoid() if sig_out else nn.Identity()
            )
        else:
            self.layers = nn.Sequential(
                nn.LayerNorm(in_dim) if norm else nn.Identity(),
                nn.Linear(in_dim, in_dim * 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(in_dim * 2, 1),
                nn.Sigmoid() if sig_out else nn.Identity()
            )

    def forward(self, x):
        if self.pool:
            x = (F.softmax(self.pool(x), dim=1).transpose(-1, -2) @ x).squeeze(-2)
        return self.layers(x).squeeze(-1)
