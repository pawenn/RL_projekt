import sys
from tqdm import tqdm
from autoencoder.carracing_dataset_class import CarRacingDataset
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from pathlib import Path
import hydra
from omegaconf import DictConfig
from hydra.utils import to_absolute_path
from .auto_encoder_model import AutoEncoder
import numpy as np

@hydra.main(config_path="../configs", config_name="autoencoder", version_base="1.3")
def train(cfg: DictConfig):
    device = cfg.device if 'device' in cfg else ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    dataset_path = to_absolute_path(cfg.dataset_path)
    full_dataset = CarRacingDataset(dataset_path, subset_fraction=cfg.subset_fraction)

    # Calculate split sizes
    val_fraction = cfg.val_set_size
    val_size = int(len(full_dataset) * val_fraction)
    train_size = len(full_dataset) - val_size

    # Split dataset
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    pin_memory = device == 'cuda'

    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=12, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=4, pin_memory=pin_memory)

    # Model, loss, optimizer 
    model = AutoEncoder(latent_dim=cfg.latent_dim, in_channels=12).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)

    for epoch in range(1, cfg.epochs + 1):
        # Training loop 
        model.train()
        running_loss = 0.0
        train_loop = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.epochs} - Train", leave=False)
        for batch in train_loop:
            batch = batch.to(device)
            optimizer.zero_grad()
            recon = model(batch)
            loss = criterion(recon, batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * batch.size(0)
            train_loop.set_postfix(loss=loss.item())
        train_loss = running_loss / train_size

        # Validation loop 
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            val_loop = tqdm(val_loader, desc=f"Epoch {epoch}/{cfg.epochs} - Val", leave=False)
            for val_batch in val_loop:
                val_batch = val_batch.to(device)
                recon = model(val_batch)
                loss = criterion(recon, val_batch)
                val_loss += loss.item() * val_batch.size(0)
                val_loop.set_postfix(loss=loss.item())
        val_loss /= val_size

        print(f"Epoch [{epoch}/{cfg.epochs}] Train Loss: {train_loss:.6f}  Val Loss: {val_loss:.6f}")

    torch.save(model.state_dict(), cfg.save_path)

    return model

if __name__ == "__main__":
    train()
