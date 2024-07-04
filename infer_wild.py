import os
import numpy as np
import argparse
from tqdm import tqdm
import imageio
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from lib.utils.tools import *
from lib.utils.learning import *
from lib.utils.utils_data import flip_data
from lib.utils.vismo import render_and_save
from lib.data.dataset_wild import WildDetDataset

from mmpose.api import MMPoseInferencer

from matplotlib import pyplot as plt
from gluoncv import model_zoo, data, utils
from gluoncv.data.transforms.pose import detector_to_alpha_pose, heatmap_to_coord_alpha_pose

def parse_args():
    parser = argparse.ArgumentParser(
        prog='PerfectRep Inference',
        description='Perform in-the-wild inference on a single image or a whole video'
    )
    parser.add_argument('--config', default='train_config.yaml', type=str, metavar='FILENAME', help='config file')
    parser.add_argument('-v', '--video', type=str, help='video path')
    parser.add_argument('-i', '--image', type=str, help='image path')
    parser.add_argument('-j', '--json_path', type=str, help='alphapose detection result json path')
    parser.add_argument('-o', '--out_path', type=str, help='output path')
    parser.add_argument('--focus', type=int, default=None, help='target person id')
    parser.add_argument('--pixel', action='store_true', help='align with pixle coordinates')