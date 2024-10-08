import json
import numpy as np
import pickle
from tqdm import tqdm

from src.utils.tools import ensure_dir

def project_to_2d(X, camera_params):
    """
    Project 3D points to 2D using the Human3.6M camera projection function.
    This is a differentiable and batched reimplementation of the original MATLAB script.

    Arguments:
    X -- 3D points in *camera space* to transform (N, *, 3)
    camera_params -- intrinsic parameteres (N, 2+2+3+2=9)
    """
    assert X.shape[-1] == 3
    assert len(camera_params.shape) == 2
    assert camera_params.shape[-1] == 9
    assert X.shape[0] == camera_params.shape[0]

    while len(camera_params.shape) < len(X.shape):
        camera_params = np.expand_dims(camera_params, axis=1)

    f = camera_params[..., :2]
    c = camera_params[..., 2:4]
    k = camera_params[..., 4:7]
    p = camera_params[..., 7:]

    XX = np.clip(X[..., :2] / X[..., 2:], -1, 1)
    r2 = np.sum(XX[..., :2]**2, axis=len(XX.shape)-1, keepdims=True)

    radial = 1 + np.sum(k * np.concatenate((r2, r2**2, r2**3),
                        axis=len(r2.shape)-1), axis=len(r2.shape)-1, keepdims=True)
    tan = np.sum(p*XX, axis=len(XX.shape)-1, keepdims=True)

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

    return np.matmul(j3d - np.array(cam_params['extrinsics']['T']), np.array(cam_params['extrinsics']['R']).T)

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
    cam_path = '%s/%s/%s/%s/camera_parameters/%s/%s.json' % (
        data_root, dataset_name, subset, subj_name, camera_name, action_name)
    j3d_path = '%s/%s/%s/%s/joints3d_25/%s.json' % (
        data_root, dataset_name, subset, subj_name, action_name)

    cam_params = read_cam_params(cam_path)

    with open(j3d_path) as f:
        j3ds = np.array(json.load(f)['joints3d_25'])

    annotations = None
    ann_path = '%s/%s/%s/%s/rep_ann.json' % (
        data_root, dataset_name, subset, subj_name)
    with open(ann_path) as f:
        annotations = json.load(f)

    return j3ds, cam_params, annotations


def preprocess_data(data_root='data', dataset_name='fit3d', test_subjects=['s08', 's09']):
    data_info_path = '%s/%s/fit3d_info.json' % (data_root, dataset_name)

    with open(data_info_path) as f:  # Load dataset info about subjects and actions
        info = json.load(f)

    camera_names = info['all_camera_names']
    train_subj = info['train_subj_names']
    actions = info['subj_to_act']

    joints_3d_labels = []  # Array of np arrays for 3d joints
    joints_2d_input = []  # Array of np arrays for 2d joint inputs
    rep_annotations = {}  # Dict holding annotation for a given source label
    source_labels = []  # Source of frame

    joints_3d_labels_test = []  # Array of np arrays for 3d joints
    joints_2d_input_test = []  # Array of np arrays for 2d joint inputs
    rep_annotations_test = {}  # Dict holding annotation for a given source label
    source_labels_test = []  # Source of frame
    actions_test = [] # Exercise for each frame

    for s in test_subjects:
        assert s in train_subj, "Test subject %s is not in dataset" % s

    for subj in tqdm(train_subj):
        subj_actions = actions[subj]  # Actions of current subject in set
        for action in subj_actions:
            for camera in camera_names:
                joints_3d, cam_params, annotations = read_data(
                    data_root, dataset_name, 'train', subj, action, camera)

                joints_3d = cam_perspective_3d(joints_3d, cam_params)
                # Extract params with distortion from dict
                intrinsic_params = np.hstack(
                    list(cam_params['intrinsics_w_distortion'].values()))
                joints_2d = project_to_2d(joints_3d, np.tile(
                    # Get 2D projections from 3D joints
                    intrinsic_params, (joints_3d.shape[0], 1)))
                try:
                    # Repetition annotations for current action and subject
                    action_annotations = annotations[action]
                except KeyError:
                    action_annotations = None
                # Label for curr batch of frames
                source = '%s_%s_%s' % (subj, action, camera)
                
                if subj in test_subjects:
                    joints_3d_labels_test.append(joints_3d)
                    joints_2d_input_test.append(joints_2d)
                    if action_annotations:
                        rep_annotations_test[source] = action_annotations
                    source_labels_test.extend([source] * joints_3d.shape[0])
                    actions_test.extend([action] * joints_3d.shape[0])
                else:
                    joints_3d_labels.append(joints_3d)
                    joints_2d_input.append(joints_2d)
                    if action_annotations:
                        rep_annotations[source] = action_annotations
                    source_labels.extend([source] * joints_3d.shape[0])

    joints_3d_labels = np.concatenate(
        joints_3d_labels, axis=0)  # Unify all into one ndarray
    joints_2d_input = np.concatenate(
        joints_2d_input, axis=0)  # Unify all into one ndarray
    source_labels = np.array(source_labels)

    joints_3d_labels_test = np.concatenate(
        joints_3d_labels_test, axis=0)  # Unify all into one ndarray
    joints_2d_input_test = np.concatenate(
        joints_2d_input_test, axis=0)  # Unify all into one ndarray
    source_labels_test = np.array(source_labels_test)

    assert joints_3d_labels.shape[0] == joints_2d_input.shape[0], "Inputs and Labels are not the same size"
    assert joints_3d_labels.shape[-1] == 3, "3D joints are not 3 dimensional"
    assert joints_2d_input.shape[-1] == 2, "2D joints are not 2 dimensional"

    assert joints_3d_labels_test.shape[0] == joints_2d_input_test.shape[0], "Test Inputs and Labels are not the same size"
    assert joints_3d_labels_test.shape[-1] == 3, "Test 3D joints are not 3 dimensional"
    assert joints_2d_input_test.shape[-1] == 2, "Test 2D joints are not 2 dimensional"

    print("Successfully Processed Data\nInputs %s\nLabels %s\nSource %s" %
          (joints_2d_input.shape, joints_3d_labels.shape, source_labels.shape))
    print("Successfully Processed Test Data\nInputs %s\nLabels %s\nSource %s" %
          (joints_2d_input_test.shape, joints_3d_labels_test.shape, source_labels_test.shape))

    # Initialize data dictionary
    data = {
        'test': {
            '2d_joint_inputs': joints_2d_input_test,
            '3d_joint_labels': joints_3d_labels_test,
            'source': source_labels_test,
            'rep_annotations': rep_annotations_test,
            'actions': actions_test
        },
        'train': {
            '2d_joint_inputs': joints_2d_input,
            '3d_joint_labels': joints_3d_labels,
            'source': source_labels,
            'rep_annotations': rep_annotations
        }
    }

    ensure_dir('%s/motion3d' % (data_root))

    file_name = 'fit3d_preprocessed_data.pkl'

    with open('%s/motion3d/%s' % (data_root, file_name), 'wb') as f:
        pickle.dump(data, f)  # Serialize data dictionary

    print('Saved processes data to %s/motion3d/%s' % (data_root, file_name))

if __name__ == '__main__':
    preprocess_data()
