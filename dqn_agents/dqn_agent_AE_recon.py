import hydra
from omegaconf import DictConfig

from utils.frame_stack_wrapper import FrameStack
from .dqn_agent import DQNAgent
from networks.decoder import make_decoder
import torch.nn.functional as F
import torch
import numpy as np
from . dqn_agent import set_seed
import gymnasium as gym
import torch.nn as nn

class DQNAgentAErecon(DQNAgent):
    def __init__(self, *args,  decoder_latent_lambda: float = 1e-6, decoder_update_freq = 1, **kwargs):
        super().__init__(*args, **kwargs)
        self.decoder_update_freq = decoder_update_freq 

        
        self.decoder_latent_lambda = decoder_latent_lambda

        self.decoder = make_decoder('pixel', self.obs_dim, self.feature_dim, self.num_encoder_decoder_layers, self.num_encoder_decoder_filters).to(self.device)
        self.decoder_optimizer = torch.optim.Adam(self.decoder.parameters(), lr=self.optimizer.param_groups[0]['lr'])
        self.encoder_optimizer = torch.optim.Adam(
                self.cnn.parameters(), lr=self.optimizer.param_groups[0]['lr']
            )
        

    def update_agent(self, training_batch):
        # unpack
        states, actions, rewards, next_states, dones, _ = zip(*training_batch)
        s = torch.tensor(np.array(states), dtype=torch.float32).to(self.device)
        a = torch.tensor(np.array(actions), dtype=torch.int64).unsqueeze(1).to(self.device)
        r = torch.tensor(np.array(rewards), dtype=torch.float32).to(self.device)
        s_next = torch.tensor(np.array(next_states), dtype=torch.float32).to(self.device)
        mask = torch.tensor(np.array(dones), dtype=torch.float32).to(self.device)

        # current Q estimates for taken actions
        z = self.cnn(s)
        pred = self.q(z).gather(1, a).squeeze(1)

        # compute TD target with frozen network
        with torch.no_grad():
            next_z = self.target_cnn(s_next)
            next_q = self.target_q(next_z).max(1)[0]
            target = r + self.gamma * next_q * (1 - mask)

        loss = nn.MSELoss()(pred, target)

        # gradient step
        self.optimizer.zero_grad()
        self.cnn_optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.cnn_optimizer.step()

        # occasionally sync target network
        if self.total_steps % self.target_update_freq == 0:
            self.target_q.load_state_dict(self.q.state_dict())
            self.target_cnn.load_state_dict(self.cnn.state_dict())

        self.total_steps += 1

        #if self.decoder is not None and step % self.decoder_update_freq == 0:

        self.update_decoder_with_recon_loss(s)
        return float(loss.item())
    
    def update_decoder_with_recon_loss(self, obs):
        self.cnn_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()
        h = self.cnn(obs)

        recon = self.decoder(h)
        
        target_obs = obs

        recon_loss = F.mse_loss(recon, target_obs)
        latent_loss = 0.5 * h.pow(2).sum(dim=1).mean() 

        ae_loss = recon_loss + self.decoder_latent_lambda * latent_loss

        ae_loss.backward()
        self.encoder_optimizer.step()
        self.decoder_optimizer.step()
        

    def save(self, path: str) -> None:
        super().save(path)
        torch.save(self.decoder.state_dict(), path + "_decoder.pt")

    def load(self, path: str) -> None:
        super().load(path)
        self.decoder.load_state_dict(torch.load(path + "_decoder.pt"))

@hydra.main(config_path="../configs/", config_name="dqn_agent_AE_recon", version_base="1.1")
def main(cfg: DictConfig):
    
    # 1) build env
    env = gym.make(cfg.env.name,  continuous=False)
    env = FrameStack(env, k=3)
    # env = gym.make(cfg.env.name, render_mode="human")
    seed=1234
    set_seed(env, seed)

    # 2) map config → agent kwargs
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    agent_kwargs = dict(
        buffer_capacity=10000,
        batch_size=32,
        lr=0.001,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_final=0.01,
        epsilon_decay=500,
        target_update_freq=1000,
        seed=seed,
        device=device,
        decoder_latent_lambda=cfg.agent.decoder_latent_lambda,
        decoder_update_freq=cfg.agent.decoder_update_freq,
    )

    # 3) instantiate & train
    agent = DQNAgentAErecon(env, **agent_kwargs)
    agent.train(cfg.train.num_frames, cfg.train.eval_interval)


if __name__ == "__main__":
    main()
