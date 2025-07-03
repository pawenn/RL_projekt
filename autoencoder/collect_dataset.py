import gymnasium as gym
import numpy as np
import os
from PIL import Image
import hydra
from omegaconf import DictConfig
from pathlib import Path
from tqdm import tqdm

@hydra.main(config_path="../configs", config_name="dataset", version_base="1.3")
def collect_frames(cfg: DictConfig):

    save_path = Path(cfg.save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    stacked_path = save_path.parent / f"{save_path.name}_stacked"
    bigfile_path = stacked_path.parent / f"{stacked_path.name}_all.npy"

    skip_collection = any(save_path.glob("frame_*.png"))
    skip_stacking = stacked_path.exists() and any(stacked_path.glob("stacked_*.npy"))

    if skip_collection:
        print(f"[!] Skipping collection: {save_path} already contains frames.")
    if skip_stacking:
        print(f"[!] Skipping stacking: {stacked_path} already contains stacked data.")
    if skip_collection and skip_stacking:
        return

    if not skip_collection:
        env = gym.make(cfg.env_name, render_mode="rgb_array")
        frame_count = 0

        print(f"[•] Starting frame collection: {cfg.num_episodes} episodes × {cfg.max_steps} steps")

        for episode in tqdm(range(cfg.num_episodes), desc="Collecting Episodes"):
            obs, _ = env.reset()
            for t in range(cfg.max_steps):
                action = env.action_space.sample()
                obs, reward, done, truncated, _ = env.step(action)
                frame = obs  # RGB array

                if cfg.save_as_images:
                    img = Image.fromarray(frame)
                    img.save(save_path / f"frame_{frame_count:06d}.png")
                else:
                    np.save(save_path / f"frame_{frame_count:06d}.npy", frame)

                frame_count += 1
                if done or truncated:
                    break

        env.close()
        print(f"[✓] Done! Total frames saved: {frame_count}")

    if not skip_stacking:
        print("[•] Building stacked dataset...")
        frame_paths = load_all_frames_sorted(save_path)

        build_stacked_dataset(
            frame_paths,
            num_episodes=cfg.num_episodes,
            max_steps=cfg.max_steps,
            stack_size=cfg.stack_size,
            save_dir=stacked_path,
     
        )
    

def load_all_frames_sorted(path: Path):
    return sorted(path.glob("frame_*.png"))


def build_stacked_dataset(frame_paths, num_episodes, max_steps, save_dir: Path, stack_size=4):
    assert len(frame_paths) >= num_episodes * stack_size, \
        "Not enough frames to cover all episodes with the desired stack size."

    print(f"[•] Building stacked dataset from {len(frame_paths)} frames")

    save_dir.mkdir(parents=True, exist_ok=True)

    frames_per_episode = len(frame_paths) // num_episodes
    episodes = [frame_paths[i * frames_per_episode: (i + 1) * frames_per_episode] for i in range(num_episodes)]

    sample_id = 0

    for ep_idx, episode in enumerate(tqdm(episodes, desc="Stacking per episode")):
        if len(episode) < stack_size:
            continue
        for i in range(len(episode) - stack_size + 1):
        
            imgs = [np.array(Image.open(p)) for p in episode[i:i + stack_size]]
            stacked_img = np.concatenate(imgs, axis=2)  # (96, 96, 12)
            stacked_img = stacked_img.astype(np.float32) / 255.0

            np.save(save_dir / f"stacked_{sample_id:06d}.npy", stacked_img)

            sample_id += 1

    print(f"[✓] Finished. Total stacked samples saved: {sample_id}")


if __name__ == "__main__":
    collect_frames()
