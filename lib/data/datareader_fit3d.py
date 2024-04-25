# Adapted from Optimizing Network Structure for 3D Human Pose Estimation (ICCV 2019) (https://github.com/CHUNYUWANG/lcn-pose/blob/master/tools/data.py)

import numpy as np
import os, sys
import random
import copy
from lib.utils.tools import read_pkl
from lib.utils.data import split_clips
random.seed(0)

class DataReaderFit3D(object):
    def __init__(self, n_frames, sample_stride, data_stride_train, data_stride_test, dt_root = 'data/motion3d', dt_file = 'fit3d_source.pkl'):
        self.gt_trainset = None # Ground truth training
        self.gt_testset = None # Ground truth test
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
        trainset = self.dt_dataset['train']['joint_2d'][::self.sample_stride, :, :2].astype(np.float32)  # [N, 17, 2]
        testset = self.dt_dataset['test']['joint_2d'][::self.sample_stride, :, :2].astype(np.float32)  # [N, 17, 2]

        return trainset, testset

    def read_3d(self):
        """
        Read 3D joint data from dataset

        Returns train and test sets ([N, 17, 3])
        """
        train_labels = self.dt_dataset['train']['joint3d_image'][::self.sample_stride, :, :3].astype(np.float32)  # [N, 17, 3]
        test_labels = self.dt_dataset['test']['joint3d_image'][::self.sample_stride, :, :3].astype(np.float32)    # [N, 17, 3]

        return train_labels, test_labels

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
        vid_list_train = self.dt_dataset['train']['source'][::self.sample_stride]  # Labels for frames of train set (subject, action, camera). shpae: (1559752,)
        vid_list_test = self.dt_dataset['test']['source'][::self.sample_stride]   # Labels for frames of test set (subject, action, camera). shpae: (566920,)

        # Calculate split IDs for training and testing data
        self.split_id_train = split_clips(vid_list_train, self.n_frames, data_stride=self.data_stride_train)
        self.split_id_test = split_clips(vid_list_test, self.n_frames, data_stride=self.data_stride_test)

        return self.split_id_train, self.split_id_test
    
    def get_sliced_data(self):
        """
        Gets the data and splits into traning and test sets

        Returns: training and test data, label pairs (N, 27, 17, 3)
        """
        train_data, test_data = self.read_2d()     # train_data (1559752, 17, 2) test_data (566920, 17, 2)
        train_labels, test_labels = self.read_3d() # train_labels (1559752, 17, 3) test_labels (566920, 17, 3)
        split_id_train, split_id_test = self.get_split_id() # Split data into individual clips
        train_data, test_data = train_data[split_id_train], test_data[split_id_test]                # (N, 27, 17, 2)
        train_labels, test_labels = train_labels[split_id_train], test_labels[split_id_test]        # (N, 27, 17, 3)

        return train_data, test_data, train_labels, test_labels