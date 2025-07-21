import torch
import torch.nn as nn
import torch.nn.functional as F


class ForwardModel(nn.Module):

    def __init__(self, latent_dim=50, action_dim=5, hidden_dim=128):
        super(ForwardModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, z, a_onehot):
        """
        z: (batch_size, latent_dim)
        a: (batch_size, action_dim)
        """
        x = torch.cat([z, a_onehot], dim=1)
        return self.fc(x)
