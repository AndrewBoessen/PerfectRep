import torch
import numpy as np
import os
import random
import copy
from torch.utils.data import Dataset, DataLoader

class ActionDataset(Dataset):
    def __init__(self, args, subset_list, data_split, action_names):
        np.random.seed(0)
        self.data_root = args.data_root
        self.subset_list = subset_list
        self.data_split = data_split
        self.action_names = action_names
        file_list_all = []
        for subset in self.subset_list:
            for action in self.action_names:
                data_path = os.path.join(self.data_root, subset, self.data_split, action)
                motion_list = sorted(os.listdir(data_path))
                for i in motion_list:
                    file_list_all.append(os.path.join(data_path, i))
            self.file_list = file_list_all

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self):
        return NotImplementedError

class PowerliftingActionDataset(ActionDataset):
    def __init__(self, args, subset_list, data_split):
        action_names = ["squat", "deadlift", "pushup"]
        super(PowerliftingActionDataset, self).__init__(args, subset_list, data_split, action_names)

    def __getitem__(self, index):
        file_path = self.file_list[index]
        motion_file = read_pkl(file_path)
        action_class = motion_file["data_label"]
        if self.data_split == "train":
            action_keypoints = motion_file["data_input"]
            action_keypoints[:,:,2] = 1 # confidence is 1
            if self.flip and random.random() > 0.5:
                action_keypoints = flip_data(action_keypoints)
        elif self.data_split == "test":
            action_keypoints = motion_file["data_input"]
            action_keypoints = crop_scale(action_keypoints)
            action_keypoints[:,:,2] = 1
        else:
            raise ValueError('Data split unknown.')
        return torch.FloatTensor(action_keypoints), torch.FloatTensor(action_class)