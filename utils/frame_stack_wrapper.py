from collections import deque
import numpy as np
import gymnasium as gym


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
        obs, reward, done, truncated, info, frames = self.env.step(action)
        # next_state, reward, done, truncated, _ = self.env.step(action)
        self._frames.append(obs)
        return self._get_obs(), reward, done, truncated, info, frames

    def _get_obs(self):
        assert len(self._frames) == self._k
        # Convert each [H, W, C] frame to [C, H, W]
        frames = [np.transpose(f, (2, 0, 1)) for f in self._frames]  # [3, 84, 84]
        return np.concatenate(list(frames), axis=0)

