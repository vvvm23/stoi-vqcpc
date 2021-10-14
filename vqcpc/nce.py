import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import HelperModule

class InfoNCE(HelperModule):
    def build(self,
            c_dim: int = 256, 
            z_dim: int = 64,
            k_steps: int = 12,
            nb_negatives: int = 10,
            subsample_window: int = 0, 
        ):
        self.c_dim = c_dim
        self.z_dim = z_dim
        self.k_steps = k_steps
        self.nb_negatives = nb_negatives
        self.subsample_window = subsample_window

        self.wk = nn.Linear(c_dim, z_dim*k_steps, bias = False) # simultaneously calculate Wc for all k steps

    def _get_negative_permutations(self, z):
        z = z.reshape(-1, self.z_dim)
        return torch.stack(
            [
                torch.index_select(z, 0, torch.randperm(z.shape[0]).to(z.device))
                for i in range(self.nb_negatives)
            ],
            dim=2,
        )

    def _get_log_bilinear(self, wc_k, z_k):
        # z_k^T @ W_k @ c_t
        # eq (3) from paper, but without exp as we use log_softmax later
        return (wc_k.unsqueeze(1) @ z_k).squeeze(1) 

    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, z, c):
        z, c = z.float(), c.float()
        k_losses = []
        k_acc = []

        fz = z
        
        if self.subsample_window:
            rand_start = np.random.randint(0, c.shape[1] - self.subsample_window)
            c = c[:, rand_start: rand_start + self.subsample_window]
            z = z[:, rand_start: rand_start + self.subsample_window]
        seq_len = z.shape[1]

        wc = self.wk(c)
        z_neg = self._get_negative_permutations(fz)

        for k in range(1, self.k_steps + 1):
            z_k = z[:, k:, :].reshape(-1, self.z_dim)
            wc_k = wc[:, :-k, (k-1) * self.z_dim : k * self.z_dim].reshape(-1, self.z_dim)

            pos_logbi = self._get_log_bilinear(wc_k, z_k.unsqueeze(-1))
            zk_neg = z_neg[z_neg.shape[0] - wc_k.shape[0]:]
            neg_logbi = self._get_log_bilinear(wc_k, zk_neg)

            logbi_scores = torch.cat([pos_logbi, neg_logbi], dim=1)
            loss = -F.log_softmax(logbi_scores, dim=1)[:, 0].mean()

            pred = logbi_scores.argmax(dim=-1)
            k_acc.append((pred == 0).float().mean())
            
            k_losses.append(loss)

        return torch.stack(k_losses).mean(), torch.stack(k_acc).mean()
