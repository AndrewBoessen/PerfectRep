import os
import torch
import torch.nn.functional as F
import numpy as np
import copy

def crop_scale(motion, scale_range=[1, 1]):
    '''
    Normalize motion data to a scale range.

    Args:
        motion (numpy array): Motion data with shape (M, T, 17, 3).
        scale_range (list, optional): Range of scales to apply. Defaults to [1, 1].

    Returns:
        numpy array: Normalized motion data with shape (M, T, 17, 3).
    '''
    result = copy.deepcopy(motion)
    valid_coords = motion[motion[..., 2]!=0][:,:2]
    if len(valid_coords) < 4:
        return np.zeros(motion.shape)
    xmin = min(valid_coords[:,0])
    xmax = max(valid_coords[:,0])
    ymin = min(valid_coords[:,1])
    ymax = max(valid_coords[:,1])
    ratio = np.random.uniform(low=scale_range[0], high=scale_range[1], size=1)[0]
    scale = max(xmax-xmin, ymax-ymin) * ratio
    if scale==0:
        return np.zeros(motion.shape)
    xs = (xmin+xmax-scale) / 2
    ys = (ymin+ymax-scale) / 2
    result[...,:2] = (motion[..., :2]- [xs,ys]) / scale
    result[...,:2] = (result[..., :2] - 0.5) * 2
    result = np.clip(result, -1, 1)
    return result

def crop_scale_3d(motion, scale_range=[1, 1]):
    '''
    Normalize 3D motion data to a scale range.

    Args:
        motion (numpy array): 3D motion data with shape (T, 17, 3).
        scale_range (list, optional): Range of scales to apply. Defaults to [1, 1].

    Returns:
        numpy array: Normalized 3D motion data with shape (T, 17, 3).
    '''
    result = copy.deepcopy(motion)
    result[:,:,2] = result[:,:,2] - result[0,0,2]
    xmin = np.min(motion[...,0])
    xmax = np.max(motion[...,0])
    ymin = np.min(motion[...,1])
    ymax = np.max(motion[...,1])
    ratio = np.random.uniform(low=scale_range[0], high=scale_range[1], size=1)[0]
    scale = max(xmax-xmin, ymax-ymin) / ratio
    if scale==0:
        return np.zeros(motion.shape)
    xs = (xmin+xmax-scale) / 2
    ys = (ymin+ymax-scale) / 2
    result[...,:2] = (motion[..., :2]- [xs,ys]) / scale
    result[...,2] = result[...,2] / scale
    result = (result - 0.5) * 2
    return result

def flip_data(data):
    """
    horizontal flip
        data: [N, F, 17, D] or [F, 17, D]. X (horizontal coordinate) is the first channel in D.
    Return
        result: same
    """
    left_joints = [4, 5, 6, 11, 12, 13]
    right_joints = [1, 2, 3, 14, 15, 16]
    flipped_data = copy.deepcopy(data)
    flipped_data[..., 0] *= -1                                               # flip x of all joints
    flipped_data[..., left_joints+right_joints, :] = flipped_data[..., right_joints+left_joints, :]
    return flipped_data

def resample(ori_len, target_len, replay=False, randomness=True):
    if replay:
        if ori_len > target_len:
            st = np.random.randint(ori_len-target_len)
            return range(st, st+target_len)  # Random clipping from sequence
        else:
            return np.array(range(target_len)) % ori_len  # Replay padding
    else:
        if randomness:
            even = np.linspace(0, ori_len, num=target_len, endpoint=False)
            if ori_len < target_len:
                low = np.floor(even)
                high = np.ceil(even)
                sel = np.random.randint(2, size=even.shape)
                result = np.sort(sel*low+(1-sel)*high)
            else:
                interval = even[1] - even[0]
                result = np.random.random(even.shape)*interval + even
            result = np.clip(result, a_min=0, a_max=ori_len-1).astype(np.uint32)
        else:
            result = np.linspace(0, ori_len, num=target_len, endpoint=False, dtype=int)
        return result

def split_clips(vid_list, n_frames, data_stride):
    """
    Splits a list of video frames into clips based on a specified number of frames per clip.

    Args:
        vid_list (list): A list of video frames.
        n_frames (int): The number of frames to include in each clip.
        data_stride (int): The stride to use when splitting the clips.

    Returns:
        list: A list of ranges, where each range represents a clip.
    """
    result = []  # Initialize an empty list to store the clips
    n_clips = 0  # Initialize a counter for the number of clips
    st = 0  # Initialize a starting index for the current clip
    i = 0  # Initialize an index for iterating over the video frames
    saved = set()  # Initialize a set to keep track of unique video frames

    while i < len(vid_list):
        i += 1
        if i - st == n_frames:
            result.append(range(st, i))  # Add the clip to the result list
            saved.add(vid_list[i - 1])  # Add the last frame of the clip to the saved set
            st = st + data_stride  # Update the starting index for the next clip
            n_clips += 1

        if i == len(vid_list):
            break

        if vid_list[i] != vid_list[i - 1]:
            if not (vid_list[i - 1] in saved):  # If the previous frame hasn't been saved yet
                resampled = resample(i - st, n_frames) + st  # Resample the clip
                result.append(resampled)  # Add the resampled clip to the result list
                saved.add(vid_list[i - 1])  # Add the previous frame to the saved set
            st = i

    return result
