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

def read_cam_params(cam_path):
    with open(cam_path) as f:
        cam_params = json.load(f)
        for key1 in cam_params:
            for key2 in cam_params[key1]:
                cam_params[key1][key2] = np.array(cam_params[key1][key2]) 
    return cam_params

def cam_perspective_3d(j3d, cam_params):
    """
    Convert 3d joints to camera's perspective

    Parameters:
        j3d: 3D joints (N, *, 3)
        cam_params: intrinsic and extrinsic camera parameters
    """
    return np.matmul(j3d - cam_params['extrinsics']['T'], cam_params['extrinsics']['R'].T)
    
def read_data(data_root, dataset_name, subset, subj_name, action_name, camera_name):
    """
    Read data for a specified subject and action

    Parameters:
        data_root: Data directory
        dataset_name: Parent directory of dataset
        sebset: train or test subset
        subj_name: subject to read data from
        action_name: action to read
        camera_name: camera

    Returns:
        3d_joins: (N, 25, 3)
        cam_params: extrinsic and intrinsic params
        annoations: frames of rep intervals
    """
    cam_path = '%s/%s/%s/%s/camera_parameters/%s/%s.json' % (data_root, dataset_name, subset, subj_name, camera_name, action_name)
    j3d_path = '%s/%s/%s/%s/joints3d_25/%s.json' % (data_root, dataset_name, subset, subj_name, action_name)

    cam_params = read_cam_params(cam_path)

    with open(j3d_path) as f:
        j3ds = np.array(json.load(f)['joints3d_25'])

    annotations = None
    ann_path = '%s/%s/%s/%s/rep_ann.json' % (data_root, dataset_name, subset, subj_name)
    with open(ann_path) as f:
        annotations = json.load(f)
    
    return j3ds, cam_params, annotations

def preprocess_data(data_root = 'data', dataset_name = 'fit3d_train'):
    data_info_path = '%s/fit3d_info.json' %(data_root)

    with open(data_info_path) as f: # Load dataset info about subjects and actions
        info = json.load(f)
    
    camera_names = info['all_camera_name']
    train_subj = info['train_subj_names']
    actions = info['subject_to_act']

    joints_3d_labels = [] # Array of np arrays for 3d joints
    joints_2d_input = [] # Array of np arrays for 2d joint inputs
    rep_annotations = {} # Dict holding annotation for a given source label
    source_labels = [] # Source of frame

    for subj in train_subj:
        subj_actions = actions[subj] # Actions of current subject in set
        for action in subj_actions:
            for camera in camera_names:
                joints_3d, cam_params, annotations = read_data(data_root, dataset_name, 'train', subj, action, camera)
                
                joints_3d = cam_perspective_3d(cam_params)
                intrinsic_params = np.concatenate(list(cam_params['intrinsics_w_distortion'].values())) # Extract params with distortion from dict
                joints_2d = project_to_2d(joints_3d, intrinsic_params) # Get 2D projections from 3D joints
                action_annotations = annotations[action] # Repetition annotations for current action and subject
                source = '%s_%s_%s' % (subj, action, camera) # Label for curr batch of frames

                joints_3d_labels.append(joints_3d)
                joints_2d_input.append(joints_2d)
                rep_annotations[source] = action_annotations
                source.extend(source * joints_3d.shape[0])
    
    joints_3d_labels = np.concatenate(joints_3d_labels) # Unify all into one ndarray
    joints_2d_input = np.concatenate(joints_2d_input) # Unify all into one ndarray
    source_labels = np.array(source_labels)

    assert joints_3d_labels.shape[-1] == joints_2d_input.shape[-1], "Inputs and Labels are not the same size"

                