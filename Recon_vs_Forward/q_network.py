from collections import OrderedDict

import torch
import torch.nn as nn


# class QFunction(nn.Module):
#     """MLP for q-function."""
#     def __init__(self, obs_dim, action_dim, hidden_dim):
#         super().__init__()
# 
#         self.trunk = nn.Sequential(
#             nn.Linear(obs_dim + action_dim, hidden_dim), nn.ReLU(),
#             nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
#             nn.Linear(hidden_dim, 1)
#         )
# 
#     def forward(self, obs, action):
#         assert obs.size(0) == action.size(0)
# 
#         obs_action = torch.cat([obs, action], dim=1)
#         return self.trunk(obs_action)
    

class QNetwork(nn.Module):
    """
    A simple MLP mapping state → Q‐values for each action.

    Architecture:
      Input → Linear(obs_dim→hidden_dim) → ReLU
            → Linear(hidden_dim→hidden_dim) → ReLU
            → Linear(hidden_dim→n_actions)
    """

    def __init__(self, obs_dim: int, n_actions: int, hidden_dim: int = 64) -> None:
        """
        Parameters
        ----------
        obs_dim : int
            Dimensionality of observation space.
        n_actions : int
            Number of discrete actions.
        hidden_dim : int
            Hidden layer size.
        """
        super().__init__()
        self.net = nn.Sequential(
            OrderedDict(
                [
                    ("fc1", nn.Linear(obs_dim, hidden_dim)),
                    ("relu1", nn.ReLU()),
                    ("fc2", nn.Linear(hidden_dim, hidden_dim)),
                    ("relu2", nn.ReLU()),
                    ("out", nn.Linear(hidden_dim, n_actions)),
                ]
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Batch of states, shape (batch, obs_dim).

        Returns
        -------
        torch.Tensor
            Q‐values, shape (batch, n_actions).
        """
        return self.net(x)

