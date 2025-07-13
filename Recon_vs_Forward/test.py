import time
from collections import deque
import numpy as np
import torch
import gymnasium as gym
from gymnasium.wrappers import AddRenderObservation, ResizeObservation, FrameStackObservation
import matplotlib.pyplot as plt

from dqn import DQNAgent, set_seed


class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        gym.Wrapper.__init__(self, env)
        self._k = k
        self._frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=((shp[2] * k,) + shp[0:2]),
            dtype=env.observation_space.dtype
        )
        # self._max_episode_steps = env._max_episode_steps

    def reset(self, seed=None):
        obs, info = self.env.reset(seed=seed)
        for _ in range(self._k):
            self._frames.append(obs)
        return self._get_obs(), info 

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        # next_state, reward, done, truncated, _ = self.env.step(action)
        self._frames.append(obs)
        return self._get_obs(), reward, done, truncated, info

    def _get_obs(self):
        assert len(self._frames) == self._k
        # Convert each [H, W, C] frame to [C, H, W]
        frames = [np.transpose(f, (2, 0, 1)) for f in self._frames]  # [3, 84, 84]
        return np.concatenate(list(frames), axis=0)


seed = 1234
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# env = gym.make("CartPole-v1", render_mode="rgb_array")
# env = AddRenderObservation(env, render_only=True)
# env = ResizeObservation(env, shape=(84, 84))
env = gym.make("CarRacing-v3", continuous=False)
print(env.observation_space)
print(env.action_space)

env = FrameStack(env, k=3)
set_seed(env, seed)

obs, _ = env.reset()
print(type(obs), obs.shape)


agent = DQNAgent(
    env=env,
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

num_frames = 500_000

agent.train(num_frames=num_frames)


