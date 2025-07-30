import gymnasium as gym

class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip):
        super().__init__(env)
        self._skip = skip
        self.frames = 0

    def step(self, action):
        total_reward = 0.0
        for _ in range(self._skip):
            state, reward, terminated, truncated, info = self.env.step(action)
            self.frames += 1
            total_reward += reward
            if terminated or truncated:
                break
        return state, total_reward, terminated, truncated, info, self.frames
    
 