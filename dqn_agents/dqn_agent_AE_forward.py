from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gymnasium as gym
import hydra
from omegaconf import DictConfig

try:
    from .dqn_agent import DQNAgent, set_seed
except Exception:
    from dqn_agent import DQNAgent, set_seed
from networks.encoder import make_encoder, PixelEncoder
from networks.forward_model import ForwardModel
from utils.frame_stack_wrapper import FrameStack


class DQNAgentAEForward(DQNAgent):

    def __init__(
        self, 
        num_conv_layers: int = 4,
        num_conv_filters: int = 32,
        forward_model_update_freq: int = 1,
        forward_latent_lambda: float = 1e-6,
        **kwargs
        ):
        super().__init__(
            **kwargs
        )

        # Add attributes specific to encoder and forward-prediction
        obs_shape = self.env.observation_space.shape
        self.forward_model_update_freq = forward_model_update_freq
        self.forward_latent_lambda = forward_latent_lambda

        self.encoder: PixelEncoder = make_encoder('pixel', obs_shape, self.feature_dim, num_conv_layers, num_conv_filters).to(self.device)
        self.target_encoder: PixelEncoder = make_encoder('pixel', obs_shape, self.feature_dim, num_conv_layers, num_conv_filters).to(self.device)
        self.forward_model = ForwardModel(self.feature_dim, self.action_dim).to(self.device)

        self.target_encoder.load_state_dict(self.encoder.state_dict())
        self.encoder_optimizer = optim.Adam(self.encoder.parameters(), lr=self.lr)
        self.forward_optimizer = optim.Adam(self.forward_model.parameters(), lr=self.lr)


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
    

    def update_agent(
        self, training_batch: List[Tuple[Any, Any, float, Any, bool, Dict]]
    ) -> float:
        """
        Perform one gradient update on a batch of transitions.

        Parameters
        ----------
        training_batch : list of transitions
            Each is (state, action, reward, next_state, done, info).

        Returns
        -------
        loss_val : float
            MSE loss value.
        """
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

        if self.total_steps % self.forward_model_update_freq == 0:
            self.update_forward_model(s, a, s_next)

        self.total_steps += 1
        return float(loss.item()), pred, target
    

    def update_forward_model(self, s, a, s_next):
        z = self.encoder(s) 
        z_next = self.encoder(s_next)

        a_onehot = F.one_hot(a.squeeze(1), self.action_dim)
        z_next_pred = self.forward_model(z, a_onehot)

        forward_loss = F.mse_loss(z_next_pred, z_next)
        latent_loss = 0.5 * z.pow(2).sum(dim=1).mean()  # encourage latent magnitudes to be small

        ae_loss = forward_loss + self.forward_latent_lambda * latent_loss

        self.encoder_optimizer.zero_grad()
        self.forward_optimizer.zero_grad()
        ae_loss.backward()
        self.encoder_optimizer.step()
        self.forward_optimizer.step()



@hydra.main(config_path="../configs/", config_name="dqn_agent_AE_forward", version_base="1.1")
def main(cfg: DictConfig):
    
    # 1) build env
    env = gym.make(cfg.env.name,  continuous=False)
    env = FrameStack(env, k=cfg.env.frame_stack)
    seed = cfg.seed
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
        feature_dim=cfg.agent.feature_dim,
        record_video=cfg.train.record_video,
        device=device,
        seed=seed,
    )

    # 3) instantiate & train
    agent = DQNAgentAEForward(
        num_conv_layers=cfg.agent.num_conv_layers,
        num_conv_filters=cfg.agent.num_conv_filters,
        forward_model_update_freq=cfg.agent.forward_model_update_freq,
        forward_latent_lambda=cfg.agent.forward_latent_lambda,
        **agent_kwargs
    )
    agent.train(cfg.train.num_frames, cfg.train.eval_interval)


if __name__ == "__main__":
    main()
