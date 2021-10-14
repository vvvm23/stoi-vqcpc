import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import HelperModule

from typing import Tuple
from math import exp

class EMAQuantizer(HelperModule):
    def build(self, 
            nb_entries: int,
            embedding_dim: int, 
            beta: float = 0.25,
            decay: float = 0.999,
            eps: float = 1e-4,
        ):
        self.beta = beta
        self.decay = decay
        self.eps = eps

        self.embedding_dim = embedding_dim
        self.nb_entries = nb_entries

        embedding = torch.Tensor(embedding_dim, nb_entries)
        embedding.uniform_(-1/nb_entries, 1/nb_entries)
        self.register_buffer("embedding", embedding)
        self.register_buffer("cluster_size", torch.zeros(nb_entries, dtype=torch.float32))
        self.register_buffer("embed_avg", embedding.clone())

    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, x: torch.FloatTensor):
        # x = self.conv_in(x.float()).permute(0,2,3,1)
        x = x.float()
        flatten = x.reshape(-1, self.embedding_dim)
        dist = (
            flatten.pow(2).sum(1, keepdim=True)
            - 2 * flatten @ self.embedding
            + self.embedding.pow(2).sum(0, keepdim=True)
        )
        _, embed_ind = (-dist).max(1)
        embed_onehot = F.one_hot(embed_ind, self.nb_entries).type(flatten.dtype)
        embed_ind = embed_ind.view(*x.shape[:-1])
        quantize = self.embed_code(embed_ind)

        if self.training:
            embed_onehot_sum = embed_onehot.sum(0)
            embed_sum = flatten.transpose(0, 1) @ embed_onehot

            self.cluster_size.data.mul_(self.decay).add_(
                embed_onehot_sum, alpha=1 - self.decay
            )
            self.embed_avg.data.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)
            n = self.cluster_size.sum()
            cluster_size = (
                (self.cluster_size + self.eps) / (n + self.nb_entries * self.eps) * n
            )
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(0)
            self.embedding.data.copy_(embed_normalized)

        diff = (quantize.detach() - x).pow(2).mean()
        quantize = x + (quantize - x).detach()

        return quantize, self.beta * diff, embed_ind

    def embed_code(self, embed_id: torch.LongTensor) -> torch.FloatTensor:
        return F.embedding(embed_id, self.embedding.transpose(0, 1))

class GumbelQuantizer(HelperModule):
    def build(self,
            nb_entries: int,
            embedding_dim: int,
            tau: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        ):
        self.embedding_dim = embedding_dim
        self.nb_entries = nb_entries
        embedding = torch.Tensor(nb_entries, embedding_dim)
        embedding.uniform_(-1/nb_entries, 1/nb_entries)
        self.register_parameter('embedding', nn.Parameter(embedding))

        self.logit_proj = nn.Linear(embedding_dim, nb_entries)

        assert len(tau) == 3, "temperature parameter must be a 3-tuple"
        self.max_tau, self.min_tau, self.tau_decay = tau
        self.tau = self.max_tau

    def update_tau(self, ts):
        self.tau = max(self.max_tau*exp(-self.tau_decay*ts), self.min_tau)

    def forward(self, x):
        N, T, _ = x.shape
        x = self.logit_proj(F.relu(x).view(-1, self.embedding_dim))
        if self.training:
            x = F.gumbel_softmax(x.float(), tau=self.tau, hard=True).type_as(x)
        else:
            x = F.one_hot(x.argmax(dim=-1), num_classes=self.nb_entries).type_as(x)
        idx = x.argmax(dim=-1)

        x = x @ self.embedding
        return x.view(N, T, -1), 0.0, idx
