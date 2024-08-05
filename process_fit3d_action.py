import os
import sys
import pickle
import numpy as np
import random
from src.utils.tools import read_pkl, ensure_dir
from src.data.datareader_fit3d import DataReaderFit3D
from tqdm import tqdm

ACTIONS = ["squat", "deadlift", "pushup"]

datareader = DataReaderFit3D(n_frames=243, sample_stride=1, data_stride_train=81, data_stride_test=243, dt_root='data/motion3d/', dt_file='fit3d_preprocessed_data.pkl')

train, test = datareader.get_split_id()
train2d, test2d = datareader.read_2d()
print(train2d.shape)
print(test2d.shape)

train_action, test_action = datareader.get_action_sliced_data("squat")
print(train_action.shape)
print(test_action.shape)