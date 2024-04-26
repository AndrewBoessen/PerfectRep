import json
import numpy as np
import os, sys
import pickle
import torch
import numpy as np

from lib.utils.tools import ensure_dir

ROOT_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), '.')

def project_to_2d(X, camera_params):
    """
    Project 3D points to 2D using the Human3.6M camera projection function.
    This is a differentiable and batched reimplementation of the original MATLAB script.
    
    Arguments:
    X -- 3D points in *camera space* to transform (N, *, 3)
    camera_params -- intrinsic parameteres (N, 2+2+3+2=9)

    https://github.com/facebookresearch/VideoPose3D/blob/main/common/camera.py
    """
    assert X.shape[-1] == 3
    assert len(camera_params.shape) == 2
    assert camera_params.shape[-1] == 9
    assert X.shape[0] == camera_params.shape[0]
    
    while len(camera_params.shape) < len(X.shape):
        camera_params = camera_params.unsqueeze(1)
        
    f = camera_params[..., :2]
    c = camera_params[..., 2:4]
    k = camera_params[..., 4:7]
    p = camera_params[..., 7:]
    
    XX = torch.clamp(X[..., :2] / X[..., 2:], min=-1, max=1)
    r2 = torch.sum(XX[..., :2]**2, dim=len(XX.shape)-1, keepdim=True)

    radial = 1 + torch.sum(k * torch.cat((r2, r2**2, r2**3), dim=len(r2.shape)-1), dim=len(r2.shape)-1, keepdim=True)
    tan = torch.sum(p*XX, dim=len(XX.shape)-1, keepdim=True)

    XXX = XX*(radial + tan) + p*r2
    
    return f*XXX + c

def split_data_by_intervals(data, intervals):
    """
    Splits a numpy ndarray into training and testing datasets based on provided intervals.

    Parameters:
        data (ndarray): The input ndarray to be split, with shape (n_samples, n_features).
        intervals (list of tuples): A list of tuples representing intervals. Each tuple should
            contain two integers representing the start and end indices of the interval.

    Returns:
        ndarray, ndarray: Two numpy ndarrays representing the training and testing datasets,
            respectively.
    """
    test_data = []
    for interval in test_intervals:
        start, end = interval
        test_data.extend(data[start:end])
    test_data = np.array(test_data)
    train_data = np.array([x for x in data if x not in test_data])
    return train_data, test_data



def convert_data(data_path = 'data/fit3d_train/train', out_dir = 'data/motion_3d', test_subjects = ['s03', 's011']):
    """
    Convert the dataset download into a dictionary and numpy arrays

    Returns:
    dictionary for train and test data
    """
    root_dir = data_path

    subjects = []
    train_split_ids = []
    test_split_ids = [] # index of test subjects data
    camera_names = set()
    camera_parameters = {}
    camera_params = []

    joints_3d = []
    source = []

    for root, dirnames, filenames in os.walk(root_dir, topdown=True): # Traverse data with dfs
        basedir = os.path.basename(os.path.dirname(root)) # Current base directory

        if not root: # Top of directory tree
            subjects = dirnames 
            # Check that test subjects are in dataset      
            assert all(subject in all_subjects for subject in test_subjects), "Test subjects are not in dataset"
        elif basedir in test_subjects:
            test_split_ids.append(len(joints_3d))
        elif basedir in subjects:
            train_split_ids.append(len(joints_3d))
        elif "joint_3d" in root:
            for file in filenames:
                with open(file, 'r') as joints:
                    data = json.load(joints)
                    joints_3d.append(data['joints']) # add joint data to array (17, 3)

                    file_path = os.path.join(root, file)
                    formatted_source = '_'.join(part.strip('./').replace('/', '_') for part in os.path.splitext(file_path)[0].split('/'))
                    source.append(formatted_source * len(data['joints'])) # Add source label for current frame
                    camera_params.append(camera_parameters[np.random.choice(list(camera_parameters.keys()))])
        elif "camera_parameters" in root:
            for dirnames in dirnames:
                camera_names.add(basedir)
        elif basedir in camera_names:
            camera_name = basedir
            for file in filenames:
                with open(file, 'r') as param_data:
                    data = json.load(param_data)
                    intr_params = data.get('intrinsics_w_distortion', {})
                    all_params = [param for sublist in intr_params.values() for param in sublist]
                    if camera_name in camera_parameters and camera_parameters[camera_name] != all_params:
                        raise AssertionError("Camera parameters do not match")
                    camera_parameters[camera_name] = all_params

    assert len(test_split_ids) == len(test_subjects), "Test subject split id missing"
    assert len(train_split_ids) + len(test_split_ids) == len(subjects), "Train subject split id missing"
    assert len(source) == len(joints_3d) == len(camera_params), "Joint, source and camera_params sizes are not equal"

    N = len(joints_3d) # Number of totla frames in dataset

    # Get interval of frames for test subjects
    test_intervals = []
    for i in test_split_ids:
        test_intervals.append((i, min((x for x in train_split_ids if x > i), key=lambda x: abs(x-i))))

    # Randomly sample 

    # Info to all frames in dataset
    joints_3d = np.array(shape=(N, 17, 3), dtype=np.float32)
    joints_2d = project_to_2d(joints_3d, camera_params)
    source = np.array(source)

    joints_3d_train, joints_3d_test = split_data_by_intervals(joints_3d, test_intervals)
    joints_2d_train, joints_2d_test = split_data_by_intervals(joints_2d, test_intervals)
    soure_train, source_test = split_data_by_intervals(source, test_intervals)

    
    return