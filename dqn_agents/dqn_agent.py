import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

from typing import Any, Dict, List, Tuple
import time
import random
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
from utils.frame_skipper_wrapper import SkipFrame
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
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    env.reset(seed=seed)

    # some spaces also support .seed()
    if hasattr(env.action_space, "seed"):
        env.action_space.seed(seed)
    if hasattr(env.observation_space, "seed"):
        env.observation_space.seed(seed)

    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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
        record_video: bool = False,
        skip_frames: int = 4,
        eval_episodes: int = 5
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
            eval_episodes,
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
        self.skip_frames = skip_frames
        self.eval_episodes = eval_episodes

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
        to_save = {}

        for name, value in vars(self).items():
            if isinstance(value, torch.nn.Module):
                to_save[f"{name}_state_dict"] = value.state_dict()
            elif isinstance(value, torch.optim.Optimizer):
                to_save[f"{name}_optimizer_state_dict"] = value.state_dict()

        torch.save(to_save, path)


    def load(self, path: str) -> None:
        """
        Load model & optimizer state from disk.

        Parameters
        ----------
        path : str
            File path.
        """
        checkpoint = torch.load(path, map_location=self.device)

        for name, value in vars(self).items():
            if isinstance(value, torch.nn.Module):
                key = f"{name}_state_dict"
                if key in checkpoint:
                    value.load_state_dict(checkpoint[key])
            elif isinstance(value, torch.optim.Optimizer):
                key = f"{name}_optimizer_state_dict"
                if key in checkpoint:
                    value.load_state_dict(checkpoint[key])



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
    

    def evaluate_policy(self, eval_env: gym.Env, eval_interval_num: int) -> float:
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
        episode_num = 0
        for _ in range(self.eval_episodes):
            state, _ = eval_env.reset()
            done = False
            episode_reward = 0.0
            episode_steps = 0

            while not done:
                action = self.predict_action(state, evaluate=True)
                state, reward, terminated, truncated, _, frames = eval_env.step(action)
                done = terminated or truncated
                episode_reward += reward
                episode_steps += 1
            
            episode_num += 1
            self.log_eval(
                    episode_steps, frames, episode_num, episode_reward, eval_interval_num, print_result=True
                )
            
            total_reward += episode_reward

        avg_reward = total_reward / self.eval_episodes
        avg_episode_length = episode_steps / self.eval_episodes
        return avg_reward, avg_episode_length


    def train(self, time_steps: int, eval_interval: int = 1000) -> None:
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
        eval_env = SkipFrame(eval_env, skip=self.skip_frames)
        eval_env = FrameStack(eval_env, k=3)
        set_seed(eval_env, self.seed)

        ep_reward = 0.0
        episode_rewards = []
        td_noise_episode = []
        steps = []
        start = time.time()
        eval_interval_num = 0

        if self.record_video:
            self.video.init(enabled=True)

        for time_step in range(1, time_steps + 1):
            action = self.predict_action(state)
            next_state, reward, done, truncated, _, frames = self.env.step(action)
            if self.record_video:
                self.video.record(self.env)
            # store and step
            self.buffer.add(state, action, reward, next_state, done or truncated, {})
            state = next_state
            ep_reward += reward

            # update if ready
            if len(self.buffer) >= self.batch_size:
                batch = self.buffer.sample(self.batch_size)
                loss, aux_loss, pred, target = self.update_agent(batch)

                # calculate TD noise as standard deviation per batch and average over episode
                td_errors = pred.detach().cpu().numpy() - target.detach().cpu().numpy()
                self.log_step(time_step, frames, len(episode_rewards), reward, loss, aux_loss, td_errors, time.time() - start, print_result=False)
                td_noise = float(np.std(td_errors))
                td_noise_episode.append(td_noise)

            if done or truncated:
                video_filename = f"episode_{len(episode_rewards)}.mp4"
                if self.record_video:
                    self.video.save(video_filename)
                    self.video.init(enabled=True)
                
                state, _ = self.env.reset()

                episode_rewards.append(ep_reward)
                steps.append(time_step)
                
                now = time.time()
                avg_td_noise = np.mean(td_noise_episode) if td_noise_episode else None
                self.print_episode_information(time_step, frames, len(episode_rewards), episode_rewards[-1], self.epsilon(), avg_td_noise, now - start)
                td_noise_episode = []
                ep_reward = 0.0
                
            # Evaluation
            if time_step % eval_interval == 0:
                avg_eval_reward, avg_eval_episode_length = self.evaluate_policy(eval_env, eval_interval_num)
                eval_interval_num += 1
                
        print("Training complete.")
        self.save(f"{self.__class__.__name__}_model_seed_{self.seed}.pt")
       
    
    def log_step(self, timestep: int, frame: int, episode: int, reward: float, loss: float, aux_loss: float, td_errors: np.ndarray, elapsed_time: float, print_result: bool = False) -> None:
        td_mean_abs = float(np.mean(np.abs(td_errors)))
        td_std = float(np.std(td_errors))
        td_max = float(np.max(td_errors))
        td_min = float(np.min(td_errors))

        row = f"{timestep};{frame};{episode};{reward:.4f};{loss:.4f};{aux_loss:.4f};{td_mean_abs:.4f};{td_std:.4f};{td_max:.4f};{td_min:.4f};{elapsed_time:.2f}"

        with open(f"{self.__class__.__name__}_training_log_seed{self.seed}.csv", "a") as f:
            if f.tell() == 0:
                f.write("Timestep;Frame;Episode;Reward;Loss;Aux-Loss;TD_mean_abs;TD_std;TD_max;TD_min;Time\n")
            f.write(row + "\n")

        if print_result:
            print(
                f"[STEP LOG] Timestep: {timestep} | Frame: {frame} | Episode: {episode} | "
                f"Reward: {reward:.4f} | Loss: {loss:.4f} | Aux-Loss: {aux_loss:.4f} | "
                f"TD_mean_abs: {td_mean_abs:.4f} | TD_std: {td_std:.4f} | TD_max: {td_max:.4f} | "
                f"TD_min: {td_min:.4f} | Time: {elapsed_time:.2f}s"
            )
    
    def print_episode_information(self, timestep: int, frame: int, episode: int, reward: float, epsilon: float, avg_td_noise: float, elapsed_time: float) -> None:
        print(
            f"[EPISODE LOG] Timestep: {timestep} | Frame: {frame} | Episode: {episode} | Reward: {reward:.4f} | "
            f"Epsilon: {epsilon:.4f} | Time: {elapsed_time:.2f}s"
        )

    def log_eval(self, timestep: int, frame: int, eval_episode: int, episode_reward: float, eval_interval_num: int, print_result: bool = False) -> None:
        row = f"{timestep};{frame};{eval_episode};{episode_reward:.4f};{eval_interval_num}"

        with open(f"{self.__class__.__name__}_eval_log_seed{self.seed}.csv", "a") as f:
            if f.tell() == 0:
                f.write("Timestep;Frame;Eval-Episode;Episode-Reward;Eval-Interval\n")
            f.write(row + "\n")

        if print_result:
            print(f"[EVAL LOG] Timestep: {timestep} | Frame: {frame} | Episode: {eval_episode} | Reward: {episode_reward:.4f} | Eval-Interval: {eval_interval_num}")


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
