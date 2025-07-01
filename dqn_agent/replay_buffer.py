import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
from pathlib import Path

from dqn_agent.dqn_agent_with_ae import DQNWithAE


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, transition):
        self.buffer.append(transition)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return (
            torch.stack(state),
            torch.tensor(action),
            torch.tensor(reward),
            torch.stack(next_state),
            torch.tensor(done),
        )

    def __len__(self):
        return len(self.buffer)