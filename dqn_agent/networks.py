from collections import OrderedDict

import torch
import torch.nn as nn
import os
from hydra.utils import get_original_cwd

from autoencoder.auto_encoder_model import AutoEncoder


class QNetworkWithAE(nn.Module):
    """
    A Q-network that uses a frozen encoder from a pretrained autoencoder
    to map input images to latent vectors, and then maps them to Q-values.
    """

    def __init__(self, latent_dim: int, n_actions: int, auto_encoder_weights_path: str):
        super().__init__()

        # Load pretrained autoencoder and extract encoder
        full_ae = AutoEncoder(latent_dim=latent_dim)
        auto_encoder_weights_path = os.path.join(get_original_cwd(), auto_encoder_weights_path)
        full_ae.load_state_dict(torch.load(auto_encoder_weights_path, map_location="cpu"))
        self.encoder = full_ae.encoder  # just the encoder part
        # Freeze encoder
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.encoder.eval()

        # Q-network MLP head (learnable)
        self.q_network = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            latent = self.encoder(x)
        return self.q_network(latent)


class QNetwork(nn.Module):
    """
    A simple CNN-based Q-network that maps raw images to Q-values.
    """

    def __init__(self, input_shape, n_actions: int):
        super().__init__()

        c, h, w = input_shape

        self.cnn = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=8, stride=4),  # (c, 96, 96) → (32, 23, 23)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),  # → (64, 10, 10)
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),  # → (64, 8, 8)
            nn.ReLU(),
            nn.Flatten()
        )

        # Compute conv output size
        with torch.no_grad():
            dummy = torch.zeros(1, *input_shape)
            conv_out_size = self.cnn(dummy).shape[1]

        self.mlp = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x / 255.0  # Normalize if input is raw image
        features = self.cnn(x)
        return self.mlp(features)

