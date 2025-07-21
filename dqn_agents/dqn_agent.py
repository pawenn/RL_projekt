import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

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
try:
    from .abstract_agent import AbstractAgent
except Exception:
    from abstract_agent import AbstractAgent
from buffer.buffers import ReplayBuffer
from networks.q_network import QNetwork
from utils.frame_stack_wrapper import FrameStack
from video.video_recorder import VideoRecorder


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
        feature_dim: int | None = None,
        device = torch.device('cpu'),
        record_video: bool = False
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
        feature_dim : int | None
            If not None, specifies input-dim of Q-Network (else obs-dim is used).
        device : device
            The device that is used (cpu or cuda).
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
        self.seed = seed
        set_seed(env, seed)
        self.device = device
        print(f"Using device: {self.device}")

        # Dimension values for Q-Networks
        self.feature_dim = feature_dim if feature_dim is not None else env.observation_space.shape[0]
        self.action_dim = env.action_space.n

        # main Q‐network and frozen target
        self.q = QNetwork(self.feature_dim, self.action_dim).to(self.device)
        self.target_q = QNetwork(self.feature_dim, self.action_dim).to(self.device)
        self.target_q.load_state_dict(self.q.state_dict())

        self.optimizer = optim.Adam(self.q.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_capacity)

        # hyperparams
        self.batch_size = batch_size
        self.lr = lr
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_final = epsilon_final
        self.epsilon_decay = epsilon_decay
        self.target_update_freq = target_update_freq

        self.total_steps = 0  # for ε decay and target sync

        #video recorder
        self.record_video = record_video
        if self.record_video:
            self.video_path = os.path.join(os.getcwd(), 'video', 'recordings')
            os.makedirs(self.video_path, exist_ok=True)
            self.video = VideoRecorder(self.video_path)

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
                qvals = self.q(t)
            action = int(torch.argmax(qvals, dim=1).item())
        else:
            # ε-greedy
            if np.random.rand() < self.epsilon():
                action = self.env.action_space.sample()
            else:
                t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    qvals = self.q(t)
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
        ):
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
        pred = self.q(s).gather(1, a).squeeze(1)

        # compute TD target with frozen network
        with torch.no_grad():
            next_q = self.target_q(s_next).max(1)[0]
            target = r + self.gamma * next_q * (1 - mask)

        loss = nn.MSELoss()(pred, target)

        # gradient step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # occasionally sync target network
        if self.total_steps % self.target_update_freq == 0:
            self.target_q.load_state_dict(self.q.state_dict())

        self.total_steps += 1

        return float(loss.item())
    

    def evaluate_policy(self, eval_env: gym.Env, num_episodes: int = 5) -> float:
        """
        Runs the policy without exploration to evaluate its performance.

        Parameters
        ----------
        num_episodes : int
            Number of episodes to average over.

        Returns
        -------
        float
            Average total reward per episode.
        """
      
        total_reward = 0.0
        episode_steps = 0
        for _ in range(num_episodes):
            state, _ = eval_env.reset()
            done = False
            episode_reward = 0.0
            
            while not done:
                action = self.predict_action(state, evaluate=True)
                state, reward, terminated, truncated, _ = eval_env.step(action)
                done = terminated or truncated
                episode_reward += reward
                episode_steps += 1
            total_reward += episode_reward

        avg_reward = total_reward / num_episodes
        avg_episode_length = episode_steps / num_episodes
        return avg_reward, avg_episode_length


    def train(self, num_frames: int, eval_interval: int = 1000) -> None:
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
        eval_env = gym.make(self.env.unwrapped.spec.id,  continuous=False)
        eval_env = FrameStack(eval_env, k=3)
        ep_reward = 0.0
        recent_rewards: List[float] = []
        episode_rewards = []
        steps = []
        log_data = []
        start = time.time()

        if self.record_video:
            self.video.init(enabled=True)

        for frame in range(1, num_frames + 1):
            action = self.predict_action(state)
            next_state, reward, done, truncated, _ = self.env.step(action)
            if self.record_video:
                self.video.record(self.env)
            # store and step
            self.buffer.add(state, action, reward, next_state, done or truncated, {})
            state = next_state
            ep_reward += reward

            # update if ready
            if len(self.buffer) >= self.batch_size:
                batch = self.buffer.sample(self.batch_size)
                loss = self.update_agent(batch)

            if done or truncated:
                video_filename = f"episode_{len(episode_rewards)}.mp4"
                if self.record_video:
                    self.video.save(video_filename)
                    self.video.init(enabled=True)
                
                state, _ = self.env.reset()

                recent_rewards.append(ep_reward)
                episode_rewards.append(ep_reward)
                steps.append(frame)
                ep_reward = 0.0
                now = time.time()
                print(f"Frame {frame}, Episode: {len(episode_rewards)}, Reward: {episode_rewards[-1]}; Time ={now - start}")
                log_entry = {
                    "frame": frame,
                    "episode": len(episode_rewards),
                    "reward": episode_rewards[-1],
                    "epsilon": self.epsilon(),
                    "time": now - start,
                    "avg_eval_reward": None,
                    "avg_eval_episode_length": None
                }
                log_data.append(log_entry)

            # Evaluation
            if frame % eval_interval == 0:
                avg_eval_reward, avg_eval_episode_length = self.evaluate_policy(eval_env, num_episodes=5)
                print(f"[EVAL] Frame {frame}: Average Eval Reward over 5 episodes: {avg_eval_reward:.2f}, Average episode length: {avg_eval_episode_length:.2f}")
                log_data[-1]["avg_eval_reward"] = round(avg_eval_reward, 2)
                log_data[-1]["avg_eval_episode_length"] = avg_eval_episode_length

        print("Training complete.")
        avg = np.mean(recent_rewards[-10:])
        now = time.time()
        print(
            f"DONE: Frame {frame}, AvgReward(10): {avg:.2f}, ε={self.epsilon():.3f}; Time ={now - start}"
        )
        df = pd.DataFrame(log_data)
        df.to_csv("training_and_eval_log.csv", index=False, sep=";")
        print("Saved training_and_eval_log.csv")
        


@hydra.main(config_path="../configs/", config_name="dqn_agent", version_base="1.1")
def main(cfg: DictConfig):
    
    # 1) build env
    env = gym.make(cfg.env.name)
    seed = cfg.seed
    set_seed(env, seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 2) map config → agent kwargs
    agent_kwargs = dict(
        buffer_capacity=cfg.agent.buffer_capacity,
        batch_size=cfg.agent.batch_size,
        lr=cfg.agent.lr,
        gamma=cfg.agent.gamma,
        epsilon_start=cfg.agent.epsilon_start,
        epsilon_final=cfg.agent.epsilon_final,
        epsilon_decay=cfg.agent.epsilon_decay,
        target_update_freq=cfg.agent.target_update_freq,
        # feature_dim=cfg.agent.feature_dim,
        device=device,
        seed=seed,
        record_video=cfg.train.record_video
    )

    # 3) instantiate & train
    agent = DQNAgent(env, **agent_kwargs)
    agent.train(cfg.train.num_frames, cfg.train.eval_interval)


if __name__ == "__main__":
    main()
