import os
import sys
import pickle
import numpy as np
import random
from src.utils.tools import read_pkl, ensure_dir
from src.data.datareader_fit3d import DataReaderFit3D
from tqdm import tqdm

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
train_data, test_data, train_labels, test_labels = datareader.get_sliced_data()

print("Training Data Info:", train_data.shape)
print("Test Data Info:", test_data.shape)
assert len(train_data) == len(train_labels)
assert len(test_data) == len(test_labels)

root = 'data/motion3d/Fit3D'
ensure_dir(root)

save_clips('train', root, train_data, train_labels)
save_clips('test', root, test_data, test_labels)