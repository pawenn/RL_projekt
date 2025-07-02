import hydra
from omegaconf import DictConfig
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Tuple

from dqn_agent.buffers import ReplayBuffer
from .utils import set_seed

from agent.abstract_agent import AbstractAgent
from dqn_agent.networks import QNetworkWithAE
import gymnasium as gym


class DQNAgentWithAE(AbstractAgent):
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
        encoder_weights_path,
        buffer_capacity: int = 10000,
        batch_size: int = 32,
        lr: float = 1e-3,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_final: float = 0.01,
        epsilon_decay: int = 500,
        target_update_freq: int = 1000,
        ae_latent_dim: int = 32,
        seed: int = 0,
        frame_stack_size=4,
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
            ae_latent_dim,
            seed,
        )
        self.env = env
        set_seed(env, seed)

        

        # main Q‐network and frozen target
        self.q = QNetworkWithAE(
            latent_dim=ae_latent_dim,
            n_actions=env.action_space.n,
            auto_encoder_weights_path=encoder_weights_path
        )
        self.target_q = QNetworkWithAE(
            latent_dim=ae_latent_dim,
            n_actions=env.action_space.n,
            auto_encoder_weights_path=encoder_weights_path
        )
        self.target_q.load_state_dict(self.q.state_dict())

        self.optimizer = optim.Adam(self.q.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_capacity)

        # This should only list the MLP header wights of the Q network
        # just to be sure the weights of the plugged in AE layers arent changed during training
        print("Trainable parameters:")
        for name, param in self.q.named_parameters():
            if param.requires_grad:
                print(f" - {name} | shape: {param.shape}")

        # hyperparams
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_final = epsilon_final
        self.epsilon_decay = epsilon_decay
        self.target_update_freq = target_update_freq

        self.total_steps = 0  # for ε decay and target sync
        self.frame_stack_size = frame_stack_size

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
        def preprocess(state: np.ndarray) -> torch.Tensor:
            t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)  # (1, H, W, C)
            if t.ndim == 4 and t.shape[-1] == 3:  # NHWC to NCHW
                t = t.permute(0, 3, 1, 2)
            return t

        if evaluate:
            # purely greedy
            t = preprocess(state)
            with torch.no_grad():
                qvals = self.q(t)
            action = int(torch.argmax(qvals, dim=1).item())
        else:
            # ε-greedy
            if np.random.rand() < self.epsilon():
                action = self.env.action_space.sample()
            else:
                t = preprocess(state)
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
        s = torch.tensor(np.array(states)).float() / 255.0

        s_next = torch.tensor(np.array(next_states)).float() / 255.0
        a = torch.tensor(np.array(actions), dtype=torch.int64).unsqueeze(1)
        r = torch.tensor(np.array(rewards), dtype=torch.float32)
        mask = torch.tensor(np.array(dones), dtype=torch.float32)

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
    
    def evaluate_policy(self, episodes=5):
        total_rewards = []
        for _ in range(episodes):
            obs, _ = self.env.reset()
            frame_buffer = deque(maxlen=self.frame_stack_size)
            for _ in range(self.frame_stack_size):
                frame = obs.transpose(2, 0, 1)
                frame_buffer.append(frame)
            state = np.concatenate(frame_buffer, axis=0)

            done = False
            truncated = False
            ep_reward = 0
            while not (done or truncated):
                action = self.predict_action(state, evaluate=True)
                obs, reward, done, truncated, _ = self.env.step(action)
                frame = obs.transpose(2, 0, 1)
                frame_buffer.append(frame)
                state = np.concatenate(frame_buffer, axis=0)
                ep_reward += reward
            total_rewards.append(ep_reward)

        return np.mean(total_rewards)

    def train(self, num_frames: int, eval_interval: int = 1000) -> None:
        """
        Run a training loop with RGB frame stacking. Stacked state has shape [12, H, W].
        """

        #init Framebuffer & rewards
        N = self.frame_stack_size
        frame_buffer = deque(maxlen=N)
        recent_rewards: List[float] = []
        ep_reward = 0.0

        #reset env and fill frame buffer
        obs, _ = self.env.reset()
        for _ in range(N):
            frame = obs.transpose(2, 0, 1)  # [H, W, 3] → [3, H, W]
            frame_buffer.append(frame)

        # helper to stack frames -> frame_stack_size * RGB, height, width -> 4 *3, 96, 96  
        def get_stacked_state():
            return np.concatenate(frame_buffer, axis=0)  # [12, H, W]

        stacked_state = get_stacked_state()

        # start Training
        for step in range(1, num_frames + 1):
            ### EXPERIENCE REPLAY ###
            action = self.predict_action(stacked_state)

            next_obs, reward, done, truncated, _ = self.env.step(action)
            next_frame = next_obs.transpose(2, 0, 1)  # [H, W, 3] → [3, H, W]
            frame_buffer.append(next_frame)
            stacked_next_state = get_stacked_state()
            
            self.buffer.add(stacked_state, action, reward, stacked_next_state, done or truncated, {})

            stacked_state = stacked_next_state
            ep_reward += reward

            ### EXPERIENCE REPLAY ###
            if len(self.buffer) >= self.batch_size:
                batch = self.buffer.sample(self.batch_size)
                loss = self.update_agent(batch)

                if step % 500 == 0:
                    print(f"[Step {step}] Loss: {loss:.4f} | Buffer size: {len(self.buffer)}")

            # Handle episode end
            if done or truncated:
                obs, _ = self.env.reset()
                for _ in range(N):
                    frame = obs.transpose(2, 0, 1)
                    frame_buffer.append(frame)
                stacked_state = get_stacked_state()

                recent_rewards.append(ep_reward)
                ep_reward = 0.0

                if len(recent_rewards) % 10 == 0:
                    avg = np.mean(recent_rewards[-10:])
                    print(f"Frame {frame}, AvgReward(10): {avg:.2f}, ε={self.epsilon():.3f}")

            # evaluation 
            
            if step % eval_interval == 0:
                eval_reward = self.evaluate_policy()
                print(f"[Eval @ Frame {step}] AvgEvalReward: {eval_reward:.2f}")
        print("Training complete.")



@hydra.main(config_path="../configs/", config_name="dqn_ae_config", version_base="1.1")
def main(cfg: DictConfig):
    # 1) build env
    env = gym.make(cfg.env.name, continuous=False)
    set_seed(env, cfg.seed)

    # 2) map config → agent kwargs
    agent_kwargs = dict(
        encoder_weights_path=cfg.agent.encoder_weights_path,
        buffer_capacity=cfg.agent.buffer_capacity,
        batch_size=cfg.agent.batch_size,
        lr=cfg.agent.learning_rate,
        gamma=cfg.agent.gamma,
        epsilon_start=cfg.agent.epsilon_start,
        epsilon_final=cfg.agent.epsilon_final,
        epsilon_decay=cfg.agent.epsilon_decay,
        target_update_freq=cfg.agent.target_update_freq,
        ae_latent_dim=cfg.agent.latent_dim,
        seed=cfg.seed,
        frame_stack_size=cfg.env.frame_stack,
    )

    # 3) instantiate & train
    agent = DQNAgentWithAE(env, **agent_kwargs)
    agent.train(cfg.train.num_frames, cfg.train.eval_interval)


if __name__ == "__main__":
    main()
