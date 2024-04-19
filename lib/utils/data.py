import os
import torch
import torch.nn.functional as F
import numpy as np
import copy

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

        if vid_list[i] != vid_list[i - 1]:  # If the current frame is different from the previous one
            if not (vid_list[i - 1] in saved):  # If the previous frame hasn't been saved yet
                resampled = resample(i - st, n_frames) + st  # Resample the clip
                result.append(resampled)  # Add the resampled clip to the result list
                saved.add(vid_list[i - 1])  # Add the previous frame to the saved set
            st = i

    return result
