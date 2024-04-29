# Adapted from Optimizing Network Structure for 3D Human Pose Estimation (ICCV 2019) (https://github.com/CHUNYUWANG/lcn-pose/blob/master/tools/data.py)

import numpy as np
import os, sys
import random
import copy
from lib.utils.tools import read_pkl
from lib.utils.data import split_clips
random.seed(0)

class DataReaderFit3D(object):
    def __init__(self, n_frames, sample_stride, data_stride_train, data_stride_test, dt_root = 'data/motion3d', dt_file = 'fit3d_preprocessed_data.pkl'):
        self.split_id_train = None # Index to split training data
        self.split_id_test = None # Index to split test data
        self.dt_dataset = read_pkl('%s/%s' % (dt_root, dt_file)) # Preprocessed dataset
        self.n_frames = n_frames # Number of frames in clip. i.e. context length
        self.sample_stride = sample_stride
        self.data_stride_train = data_stride_train
        self.data_stride_test = data_stride_test
    
    def fit3d_to_h36m(self, joints):
        """
        Convert the 25 keypoints to 17 to comply with h3.6m format

        Parameters:
            joints: (N, 25, 3)

        Returns:
            resized array (N, 17, 3)
        """
        assert joints.shape[-1] == 3, "Provided joints are not in correct format"
        assert joints.shape[-2] == 25, "Provided joints are not from fit3d"
        return joints[:, :17, :] # Remove last 8 entries in axis 1
    
    def read_2d(self):
        """
        Read 2D joint data from dataset

        Returns train and test sets ([N, 17, 3])
        """
        res_w = res_h = 900 # Camera resolution

        trainset = self.dt_dataset['train']['2d_joint_inputs'][::self.sample_stride, :, :2].astype(np.float32)  # [N, 17, 2]
        
        trainset[:, :, :] = trainset[:, :, :] / res_w * 2 - [1, res_h / res_w] # Norm [-1,1]

        # No conf provided, fill with 1.
        train_confidence = np.ones(trainset.shape)[:,:,0:1]
        trainset = np.concatenate((trainset, train_confidence), axis=2)  # [N, 17, 3]

        return trainset

    def read_3d(self):
        """
        Read 3D joint data from dataset

        Returns train and test sets ([N, 17, 3])
        """
        train_labels = self.dt_dataset['train']['3d_joint_labels'][::self.sample_stride, :, :3].astype(np.float32)  # [N, 17, 3]
        
        return train_labels

    def read_ann(self):
        """
        Read annoation data

        Returns dict of numpy arrays
        """
        ann = self.dt_dataset['train']['rep_annotations']

        return ann

    def get_split_id(self):
        """
        Gets the split IDs based on frame labels for training and testing data.

        Returns:
            tuple: A tuple containing the split IDs for training and testing data,
                or None for one or both if the corresponding video lists are missing.
        """
        if self.split_id_train is not None and self.split_id_test is not None:
            # If split IDs for both training and testing data are already calculated, return them
            return self.split_id_train, self.split_id_test

        # Extract video lists from the dataset
        vid_list_train = self.dt_dataset['train']['source'][::self.sample_stride]  # Labels for frames of train set (subject, action, camera). shpae: (N,)
        
        # Calculate split IDs for training and testing data
        self.split_id_train = split_clips(vid_list_train, self.n_frames, data_stride=self.data_stride_train)

        return self.split_id_train
    
    def get_action_sliced_data(self, action = None):
        """
        Get sliced clips for a given action. Used for classification finetuning

        Parameters:
            actions: Name of action to select

        Returns: training and test data, label pairs (N, 17, 3)
        """
        rep_annotations = self.read_ann()
        # Assert given action is valid
        assert any(action in key for key in rep_annotations.keys()), "Action is not defined in the dataset"

        train_data = self.read_2d() # train_data (N, 25, 3)
        train_data = self.fit3d_to_h36m(train_data) # (N, 17, 3)
        train_labels = self.read_3d() # train_labels (N, 25, 3)
        train_labels = self.fit3d_to_h36m(train_labels) # (N, 17, 3)
        source = self.dt_dataset['train']['source']

        actions_ids = np.where(action in source)[0]

        actions_frames_2d = train_data[actions_ids]
        actions_frames_3d = train_labels[actions_ids]

        return actions_frames_2d, actions_frames_3d 
        
    
    def get_sliced_data(self):
        """
        Gets the data and splits into traning and test sets

        Returns: training and test data, label pairs (N, 1504, 17, 3)
        """
        train_data = self.read_2d() # train_data (N, 25, 3)
        train_data = self.fit3d_to_h36m(train_data) # (N, 17, 3)
        train_labels = self.read_3d() # train_labels (N, 25, 3)
        train_labels = self.fit3d_to_h36m(train_labels) # (N, 17, 3)
        split_id_train = self.get_split_id() # Split data into individual clips
        train_data = train_data[split_id_train] # (18232, 243, 17, 3)
        train_labels = train_labels[split_id_train] # (18232, 243, 17, 3)

        return train_data, train_labels