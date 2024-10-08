# Adapted from Optimizing Network Structure for 3D Human Pose Estimation (ICCV 2019) (https://github.com/CHUNYUWANG/lcn-pose/blob/master/tools/data.py)

import numpy as np
import random
from src.utils.tools import read_pkl
from src.utils.data import split_clips, crop_scale, crop_scale_3d
random.seed(0)


class DataReaderFit3D(object):
    def __init__(
        self,
        n_frames,
        sample_stride,
        data_stride_train,
        data_stride_test,
        dt_root='data/motion3d',
        dt_file='fit3d_preprocessed_data.pkl'
    ):
        self.split_id_train = None  # Index to split training data
        self.split_id_test = None  # Index to split test data
        self.dt_dataset = read_pkl(
            '%s/%s' % (dt_root, dt_file))  # Preprocessed dataset
        self.n_frames = n_frames  # Number of frames in clip. i.e. context length
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
        return joints[:, :17, :]  # Remove last 8 entries in axis 1

    def read_2d(self):
        """
        Read 2D joint data from dataset

        Returns train and test sets ([N, 17, 3])
        """

        trainset = self.dt_dataset['train']['2d_joint_inputs'][::self.sample_stride, :, :2].astype(
            np.float32)  # [N, 17, 2]
        testset = self.dt_dataset['test']['2d_joint_inputs'][::self.sample_stride, :, :2].astype(
            np.float32)  # [N, 17, 2]

        # No conf provided, fill with 1.
        train_confidence = np.ones(trainset.shape)[:, :, 0:1]
        trainset = np.concatenate(
            (trainset, train_confidence), axis=2)  # [N, 17, 3]
        test_confidence = np.ones(testset.shape)[:, :, 0:1]
        testset = np.concatenate(
            (testset, test_confidence), axis=2)  # [N, 17, 3]
        
        # Normalize data to -1 to 1
        trainset = crop_scale(trainset)
        testset = crop_scale(testset)

        return trainset, testset

    def read_3d(self):
        """
        Read 3D joint data from dataset

        Returns train and test sets ([N, 17, 3])
        """
        train_labels = self.dt_dataset['train']['3d_joint_labels'][::self.sample_stride, :, :3].astype(
            np.float32)  # [N, 17, 3]
        test_labels = self.dt_dataset['test']['3d_joint_labels'][::self.sample_stride, :, :3].astype(
            np.float32)  # [N, 17, 3]

        # Normalize points to range
        train_labels = crop_scale_3d(train_labels)
        test_labels = crop_scale_3d(test_labels)

        # Z relative to root joint
        train_labels[:, :, 2] = train_labels[:, :, 2] - train_labels[:, 0, 2][:, np.newaxis]
        test_labels[:, :, 2] = test_labels[:, :, 2] - test_labels[:, 0, 2][:, np.newaxis]

        return train_labels, test_labels

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
        # Labels for frames of train set (subject, action, camera). shpae: (N,)
        vid_list_train = self.dt_dataset['train']['source'][::self.sample_stride]
        vid_list_test = self.dt_dataset['test']['source'][::self.sample_stride]

        # Calculate split IDs for training and testing data
        self.split_id_train = split_clips(
            vid_list_train, self.n_frames, data_stride=self.data_stride_train)
        self.split_id_test = split_clips(
            vid_list_test, self.n_frames, data_stride=self.data_stride_test
        )

        return self.split_id_train, self.split_id_test

    def get_action_split_id(self, action):
        """
        Gets the split IDs based on frame labels for training and testing for the given action
        
        Parameters:
            action: Name of action to select

        Returns:
            tuple: A tuple containing the split IDs for training and testing data,
                or None for one or both if the corresponding video lists are missing.
        """
        # Extract video lists from the dataset
        # Labels for frames of train set (subject, action, camera). shpae: (N,)
        vid_list_train = self.dt_dataset['train']['source']
        vid_list_test = self.dt_dataset['test']['source']

        action_list_train = vid_list_train[np.char.find(vid_list_train, action) >= 0][::self.sample_stride]
        action_list_test = vid_list_test[np.char.find(vid_list_test, action) >= 0][::self.sample_stride]

        action_split_id_train = split_clips(action_list_train, self.n_frames, data_stride=self.data_stride_train)
        action_split_id_test = split_clips(action_list_test, self.n_frames, data_stride=self.data_stride_train)

        return action_split_id_train, action_split_id_test

    def get_action_sliced_data(self, action=None):
        """
        Get sliced clips for a given action. Used for classification finetuning

        Parameters:
            actions: Name of action to select

        Returns: training and test data, label pairs (N, 17, 3)
        """
        rep_annotations = self.read_ann()
        # Assert given action is valid
        assert any(action in key for key in rep_annotations.keys()
                   ), "Action is not defined in the dataset"

        train_data, test_data = self.read_2d()  # train_data (N, 25, 3)
        train_data = self.fit3d_to_h36m(train_data)  # (N, 17, 3)
        test_data = self.fit3d_to_h36m(test_data) # (N, 17, 3)

        train_source = self.dt_dataset['train']['source']
        test_source = self.dt_dataset['test']['source']

        train_actions_ids = np.where(np.char.find(train_source, action) != -1)[0] # get ids of action in train set
        test_actions_ids = np.where(np.char.find(test_source, action) != -1)[0] # get ids of actions in test set
        
        train_actions_frames = train_data[train_actions_ids] # get all frames with action
        test_actions_frames = test_data[test_actions_ids] # get all frames with action

        split_id_train, split_id_test = self.get_action_split_id(action=action) # split data into clips
        train_data, test_data = train_actions_frames[split_id_train], test_actions_frames[split_id_test]  

        return train_data, test_data

    def get_sliced_data(self):
        """
        Gets the data and splits into traning and test sets

        Returns: training and test data, label pairs (N, 1504, 17, 3)
        """
        train_data, test_data = self.read_2d()  # train_data (N, 25, 3)
        train_data = self.fit3d_to_h36m(train_data)  # (N, 17, 3)
        test_data = self.fit3d_to_h36m(test_data) # (N, 17, 3)
        train_labels, test_labels = self.read_3d()  # train_labels (N, 25, 3)
        train_labels = self.fit3d_to_h36m(train_labels)  # (N, 17, 3)
        test_labels = self.fit3d_to_h36m(test_labels) # (N, 17 ,3)

        split_id_train, split_id_test = self.get_split_id()  # Split data into individual clips
        train_data, test_data = train_data[split_id_train], test_data[split_id_test]  # (18232, 243, 17, 3)
        train_labels, test_labels = train_labels[split_id_train], test_labels[split_id_test]  # (18232, 243, 17, 3)

        return train_data, test_data, train_labels, test_labels

    def denormalize(self, test_data):
        # data: (N, n_frames, 51) or data: (N, n_frames, 17, 3)        
        n_clips = test_data.shape[0]
        data = test_data.reshape([n_clips, -1, 17, 3])
        # denormalize (x,y,z) coordiantes for results
        for idx, item in enumerate(data):
            res_w, res_h = 900, 900
            data[idx, :, :, :2] = (data[idx, :, :, :2] + np.array([1, res_h / res_w])) * res_w / 2
            data[idx, :, :, 2:] = data[idx, :, :, 2:] * res_w / 2
        return data # [n_clips, -1, 17, 3]