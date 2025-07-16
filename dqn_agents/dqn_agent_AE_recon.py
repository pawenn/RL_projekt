from .dqn_agent import DQNAgent
from ..networks.decoder import make_decoder
import torch.nn.functional as F
import torch
import numpy as np

class DQNAgentAErecon(DQNAgent):
    def __init__(self, *args, recon_weight: float = 1.0, latent_l2_weight: float = 1e-6, **kwargs):
        super().__init__(*args, **kwargs)

        self.recon_weight = recon_weight
        self.latent_l2_weight = latent_l2_weight

        obs_shape = self.env.observation_space.shape
        feature_dim = self.cnn.feature_dim  

        self.decoder = make_decoder(feature_dim, obs_shape).to(self.device)
        self.decoder_optimizer = torch.optim.Adam(self.decoder.parameters(), lr=self.optimizer.param_groups[0]['lr'])

    def update_agent(self, training_batch):
        # Standard DQN update
        loss_val = super().update_agent(training_batch)

        # AE reconstruction loss
        states, _, _, _, _, _ = zip(*training_batch)
        s = torch.tensor(np.array(states), dtype=torch.float32).to(self.device)

        with torch.no_grad():
            z = self.cnn(s)

        recon = self.decoder(z)
        recon_loss = F.mse_loss(recon, s)
        latent_loss = 0.5 * z.pow(2).sum(1).mean()

        ae_loss = self.recon_weight * recon_loss + self.latent_l2_weight * latent_loss

        self.decoder_optimizer.zero_grad()
        ae_loss.backward()
        self.decoder_optimizer.step()

        return loss_val, ae_loss.item()

    def save(self, path: str) -> None:
        super().save(path)
        torch.save(self.decoder.state_dict(), path + "_decoder.pt")

    def load(self, path: str) -> None:
        super().load(path)
        self.decoder.load_state_dict(torch.load(path + "_decoder.pt"))
