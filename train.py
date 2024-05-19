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
    parser = argparse.ArgumentParser()
    # Add Arguments for Data and Hyperparameters
    parser.add_argument('-c', '--checkpoint', default=None, type=str, metavar='PATH', help='path to checkpoint binary to load (e.g. model.pt file)')
    parser.add