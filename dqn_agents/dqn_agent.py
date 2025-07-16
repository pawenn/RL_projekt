from typing import Any, Dict, List, Tuple
import time
import gymnasium as gym
import hydra
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf import DictConfig
from .abstract_agent import AbstractAgent
from buffer.buffers import ReplayBuffer
from networks.q_network import QNetwork
from utils.frame_stack_wrapper import FrameStack

from networks.encoder import make_encoder, PixelEncoder


def set_seed(env: gym.Env, seed: int = 0) -> None:
    """
    Seed Python, NumPy, PyTorch and the Gym environment for reproducibility.

    Parameters
    ----------
    env : gym.Env
        The Gym environment to seed.
    seed : int
        Random seed.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    env.reset(seed=seed)
    # some spaces also support .seed()
    if hasattr(env.action_space, "seed"):
        env.action_space.seed(seed)
    if hasattr(env.observation_space, "seed"):
        env.observation_space.seed(seed)


class DQNAgent(AbstractAgent):
    """
    Deep Q‐Learning agent with ε‐greedy policy and target network.

    Derives from AbstractAgent by implementing:
      - predict_action
      - save / load
      - update_agent
    """

    def __init__(
        self,
        env: gym.Env,
        buffer_capacity: int = 10000,
        batch_size: int = 32,
        lr: float = 1e-3,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_final: float = 0.01,
        epsilon_decay: int = 500,
        target_update_freq: int = 1000,
        seed: int = 0,
        device = 'cpu'  # new
    ) -> None:
        """
        Initialize replay buffer, Q‐networks, optimizer, and hyperparameters.

        Parameters
        ----------
        env : gym.Env
            The Gym environment.
        buffer_capacity : int
            Max experiences stored.
        batch_size : int
            Mini‐batch size for updates.
        lr : float
            Learning rate.
        gamma : float
            Discount factor.
        epsilon_start : float
            Initial ε for exploration.
        epsilon_final : float
            Final ε.
        epsilon_decay : int
            Exponential decay parameter.
        target_update_freq : int
            How many updates between target‐network syncs.
        seed : int
            RNG seed.
        """
        super().__init__(
            env,
            buffer_capacity,
            batch_size,
            lr,
            gamma,
            epsilon_start,
            epsilon_final,
            epsilon_decay,
            target_update_freq,
            seed,
        )
        self.env = env
        set_seed(env, seed)
        self.seed = seed
        self.device = device
        print(f"Using device: {self.device}")
        obs_dim = env.observation_space.shape
        n_actions = env.action_space.n
        feature_dim = 50  # new

        self.cnn: PixelEncoder = make_encoder('pixel', obs_dim, feature_dim, 4, 32).to(self.device)  # new
        self.target_cnn: PixelEncoder = make_encoder('pixel', obs_dim, feature_dim, 4, 32).to(self.device)  # new
        self.target_cnn.load_state_dict(self.cnn.state_dict())  # new

        # main Q‐network and frozen target
        self.q = QNetwork(feature_dim, n_actions).to(self.device)
        self.target_q = QNetwork(feature_dim, n_actions).to(self.device)
        self.target_q.load_state_dict(self.q.state_dict())

        self.optimizer = optim.Adam(self.q.parameters(), lr=lr)
        self.cnn_optimizer = optim.Adam(self.cnn.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_capacity)

        # hyperparams
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_final = epsilon_final
        self.epsilon_decay = epsilon_decay
        self.target_update_freq = target_update_freq

        self.total_steps = 0  # for ε decay and target sync

    def epsilon(self) -> float:
        """
        Compute current ε by exponential decay.

        Returns
        -------
        float
            Exploration rate.
        """
        return self.epsilon_final + (self.epsilon_start - self.epsilon_final) * np.exp(
            -1.0 * self.total_steps / self.epsilon_decay
        )

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

    def save(self, path: str) -> None:
        """
        Save model & optimizer state to disk.

        Parameters
        ----------
        path : str
            File path.
        """
        torch.save(
            {
                "parameters": self.q.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            },
            path,
        )

    def load(self, path: str) -> None:
        """
        Load model & optimizer state from disk.

        Parameters
        ----------
        path : str
            File path.
        """
        checkpoint = torch.load(path)
        self.q.load_state_dict(checkpoint["parameters"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])

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
        return float(loss.item())

    def train(self, num_frames: int, eval_interval: int = 1000, bin_size: int = 1000) -> None:
        """
        Run a training loop for a fixed number of frames.

        Parameters
        ----------
        num_frames : int
            Total environment steps.
        eval_interval : int
            Every this many episodes, print average reward.
        """
        state, _ = self.env.reset()
        ep_reward = 0.0
        recent_rewards: List[float] = []
        episode_rewards = []
        steps = []
        start = time.time()

        for frame in range(1, num_frames + 1):
            action = self.predict_action(state)
            next_state, reward, done, truncated, _ = self.env.step(action)

            # store and step
            self.buffer.add(state, action, reward, next_state, done or truncated, {})
            state = next_state
            ep_reward += reward

            # update if ready
            if len(self.buffer) >= self.batch_size:
                batch = self.buffer.sample(self.batch_size)
                _ = self.update_agent(batch)

            if done or truncated:
                state, _ = self.env.reset()
                recent_rewards.append(ep_reward)
                episode_rewards.append(ep_reward)
                steps.append(frame)
                ep_reward = 0.0
                # logging
                # if len(recent_rewards) % 10 == 0:
                    # avg = np.mean(recent_rewards[-10:])
                    # now = time.time()
                    # print(
                    #     f"Frame {frame}, AvgReward(10): {avg:.2f}, ε={self.epsilon():.3f}; {start=} - {now=} - diff={now - start}"
                    # )
                now = time.time()
                print(f"Frame {frame}, Episode: {len(episode_rewards)}, Reward: {episode_rewards[-1]}; {start=} - {now=} - diff={now - start}")


        print("Training complete.")
        avg = np.mean(recent_rewards[-10:])
        now = time.time()
        print(
            f"DONE: Frame {frame}, AvgReward(10): {avg:.2f}, ε={self.epsilon():.3f}; {start=} - {now=} - diff={now - start}"
        )
        # training_data = pd.DataFrame({"steps": steps, "rewards": episode_rewards, "bin": [s // bin_size for s in steps]})
        # training_data.to_csv(f"training_data_DQN_seed_{self.seed}.csv", index=False)


@hydra.main(config_path="../configs/", config_name="dqn_agent_RI", version_base="1.1")
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
        device=device
    )

    # 3) instantiate & train
    agent = DQNAgent(env, **agent_kwargs)
    agent.train(cfg.train.num_frames, cfg.train.eval_interval)


if __name__ == "__main__":
    main()
