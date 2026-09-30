"""SimCLR-style self-supervised contrastive learning for wind turbine vibration streams.

This module defines the 1D convolutional feature encoder and a placeholder
NT-Xent (Normalized Temperature-scaled Cross Entropy) loss. It ships no data
loaders, augmentation pipelines, training loops, or trained weights.
Run ``python model.py`` to build the reference configuration, inspect parameter
counts, and run a smoke-test forward pass.

The wt-pm platform imports this file as-is and uses ``ContrastiveEncoder``
inside adapter ``m08-contrastive-ssl`` (substituting a full negative-pair
NT-Xent contrastive loss during training). Keep the class name and forward
signature backwards-compatible.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveEncoder(nn.Module):
    """1D-CNN encoder and projection head for self-supervised contrastive learning.

    Parameters
    ----------
    input_dim : int, default=1024
        Nominal waveform window length in samples (e.g. 1024-sample vibration
        windows). Because an AdaptiveAvgPool1d(1) layer is used before flattening,
        the network gracefully handles any sequence length.
    proj_dim : int, default=64
        Dimensionality of the normalized projection output vector.
    """

    def __init__(self, input_dim=1024, proj_dim=64):
        super().__init__()
        self.input_dim = input_dim
        self.proj_dim = proj_dim
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 32, 7, stride=2), nn.ReLU(),
            nn.Conv1d(32, 64, 3), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.Linear(64, 128), nn.ReLU()
        )
        self.proj = nn.Linear(128, proj_dim)

    def forward(self, x):
        """Extract features and project to a unit L2-normalized embedding.

        Parameters
        ----------
        x : torch.Tensor
            Input vibration waveforms of shape ``(batch_size, 1, samples)``
            or compatible 3D tensor of float32 values.

        Returns
        -------
        torch.Tensor
            L2-normalized projection vectors of shape ``(batch_size, proj_dim)``.
        """
        h = self.encoder(x)
        z = self.proj(h)
        return F.normalize(z, dim=1)


def nt_xent_loss(z1, z2, temperature=0.5):
    """Simplified NT-Xent placeholder alignment loss for positive representation pairs.

    Parameters
    ----------
    z1 : torch.Tensor
        Normalized projection vectors from view 1, shape ``(batch_size, proj_dim)``.
    z2 : torch.Tensor
        Normalized projection vectors from view 2, shape ``(batch_size, proj_dim)``.
    temperature : float, default=0.5
        Softmax temperature hyperparameter scaling cosine similarities.

    Returns
    -------
    torch.Tensor
        Scalar positive-pair alignment loss.

    Note
    ----
    This function computes a simplified positive-pair alignment loss. The platform
    adapter ``m08-contrastive-ssl`` implements the complete contrastive NT-Xent
    loss with all negative batch pairs and cross-entropy targeting.
    """
    z = torch.cat([z1, z2], dim=0)
    sim = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2) / temperature
    # simplified NT-Xent placeholder
    return -torch.log(torch.exp(torch.diag(sim, z1.size(0))).mean())


if __name__ == "__main__":
    # Smoke test: build the reference configuration (1024 samples, 1-channel -> 64-D projection)
    # and print architecture and parameter counts. Nothing is loaded or trained here.
    model = ContrastiveEncoder(input_dim=1024, proj_dim=64)
    n_params_64 = sum(p.numel() for p in model.parameters())
    model_32 = ContrastiveEncoder(input_dim=1024, proj_dim=32)
    n_params_32 = sum(p.numel() for p in model_32.parameters())

    print(model)
    print(f"\nParameters: {n_params_64:,} (proj_dim=64) | {n_params_32:,} (proj_dim=32)")

    # Test forward pass and loss with dummy inputs (batch=2, channels=1, samples=1024)
    x1 = torch.randn(2, 1, 1024)
    x2 = torch.randn(2, 1, 1024)
    z1 = model(x1)
    z2 = model(x2)
    loss = nt_xent_loss(z1, z2, temperature=0.5)

    print(f"Input shape:              {tuple(x1.shape)}")
    print(f"Output embedding shape:   {tuple(z1.shape)} (L2 norm: {torch.norm(z1[0]).item():.4f})")
    print(f"Placeholder NT-Xent loss: {loss.item():.4f}")
