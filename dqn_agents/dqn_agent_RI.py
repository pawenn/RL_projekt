from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import hydra
from omegaconf import DictConfig

try:
    from .dqn_agent import DQNAgent, set_seed
except Exception:
    from dqn_agent import DQNAgent, set_seed
    
from networks.encoder import make_encoder, PixelEncoder
from utils.frame_skipper_wrapper import SkipFrame
from utils.frame_stack_wrapper import FrameStack


class DQNAgentRI(DQNAgent):

    def __init__(
        self,
        num_conv_layers: int = 4,
        num_conv_filters: int = 32,
        **kwargs: dict,
        ):
        """
        Initialize Raw-Image DQN-Agent using a CNN.

        Parameters
        ----------
        num_conv_layers : int
            Number of Conv2d layers in the CNN.
        num_conv_filters : int
            Number of filters per Conv2d layer in the CNN.
        """
        super().__init__(
            **kwargs
        )

        # Add CNN specific attributes
        obs_shape = self.env.observation_space.shape
        self.cnn: PixelEncoder = make_encoder('pixel', obs_shape, self.feature_dim, num_conv_layers, num_conv_filters).to(self.device)
        self.target_cnn: PixelEncoder = make_encoder('pixel', obs_shape, self.feature_dim, num_conv_layers, num_conv_filters).to(self.device)
        self.target_cnn.load_state_dict(self.cnn.state_dict())
        self.cnn_optimizer = optim.Adam(self.cnn.parameters(), lr=self.lr)  # use the same learning rate as for q-network


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
                z = self.cnn(t)
                qvals = self.q(z)
            action = int(torch.argmax(qvals, dim=1).item())
        else:
            # ε-greedy
            if np.random.rand() < self.epsilon():
                action = self.env.action_space.sample()
            else:
                t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    z = self.cnn(t)
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
        pred : torch.Tensor
        target : torch.Tensor
        """
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
        return float(loss.item()), 0, pred, target


@hydra.main(config_path="../configs/", config_name="dqn_agent_RI", version_base="1.1")
def main(cfg: DictConfig):
    
    for seed in cfg.seeds:
        # 1) build env
        env = gym.make(cfg.env.name,  continuous=False)
        env = SkipFrame(env, skip=cfg.env.skip_frames)
        env = FrameStack(env, k=cfg.env.frame_stack)
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
            skip_frames=cfg.env.skip_frames,
            seed=seed,
        )

        # 3) instantiate & train
        agent = DQNAgentRI(
            num_conv_layers=cfg.agent.num_conv_layers,
            num_conv_filters=cfg.agent.num_conv_filters,
            **agent_kwargs
        )
        agent.train(cfg.train.num_train_steps, cfg.train.eval_interval)
        env.close()


if __name__ == "__main__":
    main()

