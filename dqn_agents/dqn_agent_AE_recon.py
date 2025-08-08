from typing import Any, Dict, Tuple
import hydra
from omegaconf import DictConfig

import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

try:
    from .dqn_agent import DQNAgent, set_seed
except Exception:
    from dqn_agent import DQNAgent, set_seed

from networks.encoder import make_encoder, PixelEncoder
from networks.decoder import make_decoder
from utils.frame_skipper_wrapper import SkipFrame
from utils.frame_stack_wrapper import FrameStack

class DQNAgentAErecon(DQNAgent):

    def __init__(
        self, 
        num_conv_layers: int = 4,
        num_conv_filters: int = 32,
        decoder_latent_lambda: float = 1e-6, 
        decoder_update_freq = 1, 
        **kwargs
        ):
        """
        Initialize Reconstruction DQN-Agent

        Parameters
        ----------
        num_conv_layers : int
            Number of Conv2d layers in the Encoder.
        num_conv_filters : int
            Number of filters per Conv2d layer in the Encoder.
        decoder_latent_lambda : float
            Update weight for L2 penalty on latent representation
        decoder_update_freq : int
            After how many update-steps of the target-network the decoder and encoder are updated using reconstruction
        """
        super().__init__(**kwargs)
        self.decoder_update_freq = decoder_update_freq
        obs_shape = self.env.observation_space.shape
        self.encoder: PixelEncoder = make_encoder('pixel', obs_shape, self.feature_dim, num_conv_layers, num_conv_filters).to(self.device)
        self.target_encoder: PixelEncoder = make_encoder('pixel', obs_shape, self.feature_dim, num_conv_layers, num_conv_filters).to(self.device)
        self.target_encoder.load_state_dict(self.encoder.state_dict())
        self.encoder_optimizer = optim.Adam(self.encoder.parameters(), lr=self.lr) 

        
        self.decoder_latent_lambda = decoder_latent_lambda

        self.decoder = make_decoder('pixel', obs_shape, self.feature_dim, num_conv_layers, num_conv_filters).to(self.device)
        self.decoder_optimizer = torch.optim.Adam(self.decoder.parameters(), lr=self.optimizer.param_groups[0]['lr'])
        
        
    def predict_action(
        self, state: np.ndarray, info: Dict[str, Any] = {}, evaluate: bool = False
    ) -> Tuple[int, Dict]:
        """
        Choose action via ε‐greedy (or purely greedy in eval mode).

        Parameters
        ----------
        state : np.ndarray 
            Current observation.
        info : dict
            Gym info dict (unused here).
        evaluate : bool
            If True, always pick argmax(Q).

        Returns
        -------
        action : int
        info_out : dict
            Empty dict (compatible with interface).
        """
        if evaluate:
            # purely greedy
            t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            with torch.no_grad():
                z = self.encoder(t)
                qvals = self.q(z)
            action = int(torch.argmax(qvals, dim=1).item())
        else:
            # ε-greedy
            if np.random.rand() < self.epsilon():
                action = self.env.action_space.sample()
            else:
                t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    z = self.encoder(t)
                    qvals = self.q(z)
                action = int(torch.argmax(qvals, dim=1).item())

        return action
        

    def update_agent(self, training_batch):
        # unpack
        states, actions, rewards, next_states, dones, _ = zip(*training_batch)
        s = torch.tensor(np.array(states), dtype=torch.float32).to(self.device)
        a = torch.tensor(np.array(actions), dtype=torch.int64).unsqueeze(1).to(self.device)
        r = torch.tensor(np.array(rewards), dtype=torch.float32).to(self.device)
        s_next = torch.tensor(np.array(next_states), dtype=torch.float32).to(self.device)
        mask = torch.tensor(np.array(dones), dtype=torch.float32).to(self.device)

        # current Q estimates for taken actions
        z = self.encoder(s)
        pred = self.q(z).gather(1, a).squeeze(1)

        # compute TD target with frozen network
        with torch.no_grad():
            next_z = self.target_encoder(s_next)
            next_q = self.target_q(next_z).max(1)[0]
            target = r + self.gamma * next_q * (1 - mask)

        loss = nn.MSELoss()(pred, target)

        # gradient step
        self.optimizer.zero_grad()
        self.encoder_optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.encoder_optimizer.step()

        # occasionally sync target network
        if self.total_steps % self.target_update_freq == 0:
            self.target_q.load_state_dict(self.q.state_dict())
            self.target_encoder.load_state_dict(self.encoder.state_dict())

        self.total_steps += 1

        #if self.decoder is not None and step % self.decoder_update_freq == 0:

        aux_loss = self.update_decoder_with_recon_loss(s)
        return float(loss.item()), aux_loss, pred, target
    
    def update_decoder_with_recon_loss(self, obs):
        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()
        h = self.encoder(obs)

        recon = self.decoder(h)
        
        target_obs = obs

        recon_loss = F.mse_loss(recon, target_obs)
        latent_loss = 0.5 * h.pow(2).sum(dim=1).mean() 

        ae_loss = recon_loss + self.decoder_latent_lambda * latent_loss

        ae_loss.backward()
        self.encoder_optimizer.step()
        self.decoder_optimizer.step()

        return float(ae_loss.item())
        

    def save(self, path: str) -> None:
        super().save(path)
        torch.save(self.decoder.state_dict(), path + "_decoder.pt")

    def load(self, path: str) -> None:
        super().load(path)
        self.decoder.load_state_dict(torch.load(path + "_decoder.pt"))

@hydra.main(config_path="../configs/", config_name="dqn_agent_AE_recon", version_base="1.1")

def main(cfg: DictConfig):

    for seed in cfg.seeds:
        # 1) build env
        # env = gym.make(cfg.env.name,  continuous=False, render_mode="rgb_array")

        env = gym.make(cfg.env.name,  continuous=False)
        env = SkipFrame(env, skip=cfg.env.skip_frames)
        env = FrameStack(env, k=3)
        set_seed(env, seed)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 2) map config → agent kwargs

        agent_kwargs = dict(
            env=env,
            buffer_capacity=cfg.agent.buffer_capacity,
            batch_size=cfg.agent.batch_size,
            lr=cfg.agent.lr,
            gamma=cfg.agent.gamma,
            epsilon_start=cfg.agent.epsilon_start,
            epsilon_final=cfg.agent.epsilon_final,
            epsilon_decay=cfg.agent.epsilon_decay,
            target_update_freq=cfg.agent.target_update_freq,
            skip_frames=cfg.env.skip_frames,
            seed=seed,
            device=device,
            feature_dim=cfg.agent.feature_dim,
            record_video=cfg.train.record_video,
            eval_episodes=cfg.train.eval_episodes
        )


        # 3) instantiate & train

        agent = DQNAgentAErecon(
            decoder_latent_lambda=cfg.agent.decoder_latent_lambda,
            decoder_update_freq=cfg.agent.decoder_update_freq,
            **agent_kwargs
        )
        agent.train(cfg.train.num_train_steps, cfg.train.eval_interval)
        env.close()


if __name__ == "__main__":
    main()
