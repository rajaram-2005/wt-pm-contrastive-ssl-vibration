# SimCLR/Contrastive Learning setup
import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveEncoder(nn.Module):
    def __init__(self, input_dim=1024, proj_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 32, 7, stride=2), nn.ReLU(),
            nn.Conv1d(32, 64, 3), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.Linear(64, 128), nn.ReLU()
        )
        self.proj = nn.Linear(128, proj_dim)

    def forward(self, x):
        h = self.encoder(x)
        z = self.proj(h)
        return F.normalize(z, dim=1)

def nt_xent_loss(z1, z2, temperature=0.5):
    z = torch.cat([z1, z2], dim=0)
    sim = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2) / temperature
    # simplified NT-Xent placeholder
    return -torch.log(torch.exp(torch.diag(sim, z1.size(0))).mean())
