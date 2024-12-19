import os
import cv2
import numpy as np
import torch
import pickle
from typing import Sequence
from torch.utils.data import Dataset


class LSBMPDataset(Dataset):
    """LSBMP dataset. Read images and actions.

    Args:
        data_dir (str): path to images folder
    """

    def __init__(
            self,
            data_dir
    ):
        self.data_dir = data_dir
        self.ids = os.listdir(data_dir)
        self.ids.sort()
        self.traj_length = len(os.listdir(os.path.join(self.data_dir, self.ids[0]))) - 2
        self.empties = []
        self.actions = []
        self.states = []
        # Preload everything to memory, ~1GB for 10000 envs
        for id in self.ids:
            with open(os.path.join(self.data_dir, id, "empty.bin"), 'rb') as fp:
                self.empties.append(pickle.load(fp))
            with open(os.path.join(self.data_dir, id, "actions.bin"), 'rb') as fp:
                self.actions.append(pickle.load(fp))
            states = []
            for i in range(self.traj_length):
                with open(os.path.join(self.data_dir, id, f"state_{i}.bin"), 'rb') as fp:
                    states.append(pickle.load(fp))
            self.states.append(states)

    def __getitem__(self, i):
        env_index = i // self.traj_length
        action_index = i % self.traj_length
        x_t = self.states[env_index][action_index]
        x_tplus = self.states[env_index][action_index + 1]
        x_empty = self.empties[env_index]
        u_t = self.actions[env_index][action_index]

        return torch.tensor(x_t, dtype=torch.float32), \
                torch.tensor(x_tplus, dtype=torch.float32), \
                torch.tensor(x_empty, dtype=torch.float32), \
                torch.tensor(u_t, dtype=torch.float32)

    def __len__(self):
        return len(self.ids) * self.traj_length
