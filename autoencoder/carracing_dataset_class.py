import torch
from torch.utils.data import Dataset
import numpy as np
import os
import math
class CarRacingDataset(Dataset):
    def __init__(self, data_dir, transform=None, subset_fraction=1.0):
        """
        Args:
            data_dir (str): Path to directory with .npy files.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data_dir = data_dir
        all_files = [f for f in os.listdir(data_dir) if f.endswith('.npy')]
        # Shuffle to get random subset
        all_files.sort()  # stable order, or random.shuffle(all_files) if you want randomness
        subset_size = math.ceil(len(all_files) * subset_fraction)
        self.file_names = all_files[:subset_size]
        self.transform = transform

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        file_path = os.path.join(self.data_dir, self.file_names[idx])
        data = np.load(file_path)  # shape (96, 96, 12)

        data = torch.from_numpy(data).float().permute(2, 0, 1)  # (12, 96, 96)

        if self.transform:
            data = self.transform(data)

        return data
