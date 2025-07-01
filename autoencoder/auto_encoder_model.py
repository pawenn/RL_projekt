import torch
import torch.nn as nn

class AutoEncoder(nn.Module):
    def __init__(self, latent_dim=32, in_channels=12):
        super().__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=4, stride=2, padding=1),  # 96 -> 48
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1),  # 48 -> 24
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # 24 -> 12
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 12 * 12, latent_dim)
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64 * 12 * 12),
            nn.ReLU(),
            nn.Unflatten(1, (64, 12, 12)),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # 12 -> 24
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),  # 24 -> 48
            nn.ReLU(),
            nn.ConvTranspose2d(16, in_channels, kernel_size=4, stride=2, padding=1),   # 48 -> 96
            nn.Sigmoid()
        )

    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon

    def encode(self, x):
        return self.encoder(x)
