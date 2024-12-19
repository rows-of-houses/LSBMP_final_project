import os
import cv2
import numpy as np
import torch
import pickle
from typing import Sequence
from torch.utils.data import Dataset


class CollisionDataset(Dataset):
    """Collision dataset. Read images and labels.

    Args:
        data_dir (str): path to images folder
    """

    def __init__(
            self,
            data_dir
    ):
        self.data_dir = data_dir
        with open(os.path.join(self.data_dir, "latents1.bin"), 'rb') as fp:
            self.latents1 = pickle.load(fp)
        with open(os.path.join(self.data_dir, "latents2.bin"), 'rb') as fp:
            self.latents2 = pickle.load(fp)
        with open(os.path.join(self.data_dir, "labels.bin"), 'rb') as fp:
            self.labels = pickle.load(fp)
        self.ids = os.listdir(data_dir)
        self.num_envs = len(self.labels)
        self.num_pairs = len(self.labels[1])
        self.empties = []
        # Preload everything to memory
        for i in range(self.num_envs):
            with open(os.path.join(self.data_dir, f"empty_{i}.bin"), 'rb') as fp:
                self.empties.append(pickle.load(fp))

    def __getitem__(self, i):
        env_index = i // self.num_pairs
        pair_index = i % self.num_pairs
        # Simple augmentation, 2x more data
        reverse = False
        if env_index >= self.num_envs:
            env_index -= self.num_envs
            reverse = True
        z1 = self.latents1[env_index][pair_index]
        z2 = self.latents2[env_index][pair_index]
        if reverse:
            z1, z2 = z2, z1
        x_empty = self.empties[env_index]
        label = self.labels[env_index][pair_index]

        return torch.tensor(z1, dtype=torch.float32), \
                torch.tensor(z2, dtype=torch.float32), \
                torch.tensor(x_empty, dtype=torch.float32), \
                float(label)

    def __len__(self):
        return self.num_envs * self.num_pairs * 2
