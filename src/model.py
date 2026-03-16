import torch
import torch.nn as nn
import numpy as np


class FourierEmbedding(nn.Module):
    """
    Random Fourier feature embedding to help PINNs learn high-frequency
    spatial patterns (boundary layers, pressure gradients near airfoil).
    """
    def __init__(self, in_dim=2, embed_dim=64, scale=1.0):
        super().__init__()
        B = torch.randn(in_dim, embed_dim) * scale
        self.register_buffer("B", B)

    def forward(self, x):
        proj = x @ self.B  # (N, embed_dim)
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)  # (N, 2*embed_dim)


class ResidualBlock(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(width, width),
            nn.Tanh(),
            nn.Linear(width, width),
        )
        self.act = nn.Tanh()

    def forward(self, x):
        return self.act(x + self.net(x))


class PINN(nn.Module):
    """
    Physics-Informed Neural Network for 2D steady incompressible Navier-Stokes.

    Architecture:
    - Fourier feature embedding (captures multi-scale spatial features)
    - Deep MLP with residual connections (prevents vanishing gradients)
    - Output: [u, v, p]
    """
    def __init__(self, fourier_embed_dim=64, fourier_scale=1.0, width=256, depth=6):
        super().__init__()

        self.embedding = FourierEmbedding(2, fourier_embed_dim, fourier_scale)
        input_dim = 2 * fourier_embed_dim

        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, width),
            nn.Tanh(),
        )

        self.res_blocks = nn.ModuleList(
            [ResidualBlock(width) for _ in range(depth)]
        )

        self.output_layer = nn.Linear(width, 3)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        z = self.embedding(x)
        z = self.input_layer(z)
        for block in self.res_blocks:
            z = block(z)
        return self.output_layer(z)
