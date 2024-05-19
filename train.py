import os
import numpy as np
import argparse
import errno
import math
import pickle
import tensorboardX
from tqdm import tqdm
from time import time
import copy
import random
import prettytable

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

def parse_args():
    parser = argparse.ArgumentParser(
        prog='PerfectRep Train',
        description='A script for pretraining the 3D pose estimation model. This trains the model on the h3.6m and optional fit3d dataset'
    )
    # Add Arguments for Data and Hyperparameters
    parser.add_argument('-d', '--data_path', default='data/', type=str, metavar='PATH', help='path to training data directory')
    parser.add_argument('-f', '--fit3d', default=False, type=bool, help='use fit3d data for training')
    parser.add_argument('-c', '--checkpoint', default='', type=str, metavar='FILENAME', help='filename of checkpoint binary to load (e.g. model.pt file)')
    parser.add_argument('-e', '--evaluate', default=False, type=bool, help='evaluate accuracy of model after each epoch')
    parser.add_argument('-b', '--batch_size', default=1, type=int, help='batch size for training')
    parser.add_argument('-e', '--epochs', default=10, type=int, help='number of epochs to train for')
    
def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)