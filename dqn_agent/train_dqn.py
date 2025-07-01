import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter
from omegaconf import DictConfig
import hydra

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


@hydra.main(config_path="../configs", config_name="dqn_ae_config", version_base="1.3")
def train(cfg: DictConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # === Environment ===
    env = gym.make(cfg.env_name, render_mode="rgb_array")
    obs, _ = env.reset()
    obs = preprocess_obs(obs)

    # === Model and Target Network ===
    model = DQNWithAE(
        latent_dim=cfg.latent_dim,
        num_actions=env.action_space.n,
        encoder_weights_path=cfg.encoder_weights_path
    ).to(device)

    target_model = DQNWithAE(
        latent_dim=cfg.latent_dim,
        num_actions=env.action_space.n,
        encoder_weights_path=cfg.encoder_weights_path
    ).to(device)
    target_model.load_state_dict(model.state_dict())

    optimizer = optim.Adam(model.q_network.parameters(), lr=cfg.lr)
    criterion = nn.MSELoss()
    replay_buffer = ReplayBuffer(cfg.replay_buffer_size)
    writer = SummaryWriter()

    # === Epsilon-greedy setup ===
    epsilon = cfg.start_epsilon
    epsilon_decay = (cfg.start_epsilon - cfg.end_epsilon) / cfg.epsilon_decay_steps
    step_count = 0
    episode = 0

    while step_count < cfg.total_timesteps:
        obs, _ = env.reset()
        obs = preprocess_obs(obs)
        episode_reward = 0
        done = False

        while not done:
            step_count += 1

            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    q_values = model(obs.unsqueeze(0).to(device))
                    action = torch.argmax(q_values).item()

            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_obs = preprocess_obs(next_obs)
            replay_buffer.push((obs, action, reward, next_obs, float(done)))
            obs = next_obs
            episode_reward += reward

            # Epsilon decay
            if epsilon > cfg.end_epsilon:
                epsilon -= epsilon_decay

            # === Training step ===
            if len(replay_buffer) >= cfg.batch_size:
                states, actions, rewards, next_states, dones = replay_buffer.sample(cfg.batch_size)
                states = states.to(device)
                next_states = next_states.to(device)
                actions = actions.to(device)
                rewards = rewards.to(device)
                dones = dones.to(device)

                q_values = model(states).gather(1, actions.unsqueeze(1)).squeeze()
                with torch.no_grad():
                    target_q_values = target_model(next_states).max(1)[0]
                targets = rewards + cfg.gamma * target_q_values * (1 - dones)

                loss = criterion(q_values, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # === Target update ===
            if step_count % cfg.target_update_freq == 0:
                target_model.load_state_dict(model.state_dict())

            if step_count % cfg.log_interval == 0:
                print(f"[Step {step_count}] Reward: {episode_reward:.2f}, Epsilon: {epsilon:.3f}")
                writer.add_scalar("train/episode_reward", episode_reward, step_count)
                writer.add_scalar("train/epsilon", epsilon, step_count)

        episode += 1

    # === Save Model ===
    Path(cfg.save_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), cfg.save_path)
    print(f"[✓] DQN model with AE saved to {cfg.save_path}")
    env.close()


def preprocess_obs(obs):
    from torchvision import transforms
    import PIL.Image
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((96, 96)),
        transforms.ToTensor()
    ])
    return transform(obs).float()


if __name__ == "__main__":
    train()
