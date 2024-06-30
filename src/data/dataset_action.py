import torch
import numpy as np
import os
import random
import copy
from torch.utils.data import Dataset, DataLoader

class ActionClassifierDataset(Dataset):
    def __init__(self, datareader, n_frames=243, scale_range=[1,1], check_split=True):
        annotations = datareader.read_ann() # Annotations for rep frames
        actions = annotations.keys() # Set of all action names

        motions = []
        labels = []

        for action in actions:
            actions_frames, _ = datareader.get_action_sliced_data(action=action)
            motions.append(actions_frames)
            labels.append(action)
        self.motions = np.array(motions)
        self.labels = np.array(labels)

    def __len__(self):
        return len(self.motions) # Number of samples across all actions
    
    def __getitem__(self, idx):
        raise NotImplementedError