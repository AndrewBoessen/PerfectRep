import os
import sys
import pickle
import numpy as np
import random
from src.utils.tools import read_pkl, ensure_dir
from src.data.datareader_fit3d import DataReaderFit3D
from tqdm import tqdm

ACTIONS = ["squat", "deadlift", "pushup"]

def save_clips(dataset_name, root_path, train_data, train_labels):
    len_train = len(train_data)
    save_path = os.path.join(root_path, dataset_name)
    ensure_dir(save_path)
    for i in tqdm(range(len_train)):
        data_input, data_label = train_data[i], train_labels[i]
        data_dict = {
            "data_input": data_input,
            "data_label": data_label
        }
        with open(os.path.join(save_path, "%08d.pkl" % i), "wb") as curr_clip:  
            pickle.dump(data_dict, curr_clip)

datareader = DataReaderFit3D(n_frames=243, sample_stride=1, data_stride_train=81, data_stride_test=243, dt_root='data/motion3d/', dt_file='fit3d_preprocessed_data.pkl')

train, test = datareader.get_split_id()
train2d, test2d = datareader.read_2d()

print("Training Data Info:", train2d.shape)
print("Test Data Info:", test2d.shape)

root = 'data/action/Fit3D'
ensure_dir(root)

for action in ACTIONS:
    print(f"{action} Info:")
    train_action, test_action = datareader.get_action_sliced_data(action=action)
    print(train_action.shape)
    print(test_action.shape)

    one_hot_encode = np.zeros(len(ACTIONS), dtype=int)
    one_hot_encode[ACTIONS.index(action)] = 1
    print(one_hot_encode)

    C,F,_,_ = train_action.shape
    train_labels = np.full(shape=(C,F,len(ACTIONS)), fill_value=one_hot_encode)
    C,F,_,_ = test_action.shape
    test_labels = np.full(shape=(C,F,len(ACTIONS)), fill_value=one_hot_encode)

    assert len(train_action) == len(train_labels)
    assert len(test_action) == len(test_labels)

    save_clips(f'train/{action}', root, train_action, train_labels)
    save_clips(f'test/{action}', root, test_action, test_labels)