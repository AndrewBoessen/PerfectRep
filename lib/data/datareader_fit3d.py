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
    
    def read_2d(self):
        """
        Read 2D joint data from dataset

        Returns train and test sets ([N, 17, 3])
        """
        res_w, res_h = 900 # Camera resolution

        trainset = self.dt_dataset['train']['joint_2d'][::self.sample_stride, :, :2].astype(np.float32)  # [N, 17, 2]
        
        trainset[idx, :, :] = trainset[idx, :, :] / res_w * 2 - [1, res_h / res_w] # Norm [-1,1]

        # No conf provided, fill with 1.
        train_confidence = np.ones(trainset.shape)[:,:,0:1]
        trainset = np.concatenate((trainset, train_confidence), axis=2)  # [N, 17, 3]

        return trainset

    def read_3d(self):
        """
        Read 3D joint data from dataset

        Returns train and test sets ([N, 17, 3])
        """
        train_labels = self.dt_dataset['train']['joint3d_image'][::self.sample_stride, :, :3].astype(np.float32)  # [N, 17, 3]

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
    
    def get_sliced_data(self):
        """
        Gets the data and splits into traning and test sets

        Returns: training and test data, label pairs (N, 27, 17, 3)
        """
        train_data = self.read_2d()     # train_data (N, 25, 2)
        train_labels = self.read_3d() # train_labels (N, 25, 3)
        split_id_train = self.get_split_id() # Split data into individual clips
        train_data = train_data[split_id_train] # (N, 8*47*4=1504, 17, 2)
        train_labels, test_labels = train_labels[split_id_train] # (N, 1504, 17, 3)

        return train_data, train_labels