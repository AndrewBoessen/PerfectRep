import os
import numpy as np
import argparse
import errno
import math
import pickle
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
from torch.utils.tensorboard import SummaryWriter

from src.utils.tools import *
from src.utils.learning import *
from src.utils.data import flip_data
from src.data.dataset_motion_2d import PoseTrackDataset2D, InstaVDataset2D
from src.data.dataset_motion_3d import MotionDataset3D
from src.data.augmentation import Augmenter2D
from src.data.datareader_h36m import DataReaderH36M  
from src.model.loss import *

def parse_args():
    parser = argparse.ArgumentParser(
        prog='PerfectRep Train',
        description='A script for pretraining the 3D pose estimation model. This trains the model on the h3.6m and optional fit3d dataset'
    )
    # Add Arguments for Data and Hyperparameters
    parser.add_argument('--config', default='train_config.yaml', type=str, metavar='FILENAME', help='config file')
    parser.add_argument('-s', '--save_path', default='save/', type=str, metavar='PATH', help='path to store logs and checkpoint saves')
    parser.add_argument('-d', '--data_path', default='data/', type=str, metavar='PATH', help='path to training data directory')
    parser.add_argument('-f', '--fit3d', default=False, action='store_true', help='use fit3d data for training')
    parser.add_argument('-c', '--checkpoint', default='', type=str, metavar='FILENAME', help='filename of checkpoint binary to load (e.g. model.pt file)')
    parser.add_argument('-v', '--evaluate', default=False, action='store_true', help='evaluate accuracy of model after each epoch')
    parser.add_argument('-b', '--batch_size', default=1, type=int, help='batch size for training')
    parser.add_argument('-e', '--epochs', default=10, type=int, help='number of epochs to train for')
    args = parser.parse_args()
    return args

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def save_checkpoint(chk_path, epoch, lr, optimizer, model, min_loss):
    '''
    Save checkpoint of training state for current epoch.

    Args:
        chk_path: path to save checkpoint binary to
        epoch: current epoch to save
        lr: learning rate
        optimzer: optimizer used for training
        model: model state to save
        min_loss: minimum loss acheived during training
    '''
    assert os.path.exists(chk_path), "Error saving checkpoint: File %s path does not exist!" % cp_path
    print('Saving checkpoint to', chk_path)
    torch.save({
        'epoch': epoch + 1,
        'lr': lr,
        'optimizer': optimizer.state_dict(),
        'model': model.state_dict(),
        'min_loss' : min_loss
    }, chk_path)

def evaluate(cfg, model, test_loader, datareader):
    '''
    Copied from [MotionBERT: A Unified Perspective on Learning Human Motion Representations]
                (https://github.com/Walter0807/MotionBERT/blob/main/train.py)
    '''
    print('INFO: Evaluating Model')
    results_all = []
    model.eval()            
    with torch.no_grad():
        for batch_input, batch_gt in tqdm(test_loader):
            N, T = batch_gt.shape[:2]
            if torch.cuda.is_available():
                batch_input = batch_input.cuda()
            if cfg.no_conf:
                batch_input = batch_input[:, :, :, :2]
            if cfg.flip:    
                batch_input_flip = flip_data(batch_input)
                predicted_3d_pos_1 = model(batch_input)
                predicted_3d_pos_flip = model(batch_input_flip)
                predicted_3d_pos_2 = flip_data(predicted_3d_pos_flip)                   # Flip back
                predicted_3d_pos = (predicted_3d_pos_1+predicted_3d_pos_2) / 2
            else:
                predicted_3d_pos = model(batch_input)
            if cfg.rootrel:
                predicted_3d_pos[:,:,0,:] = 0     # [N,T,17,3]
            else:
                batch_gt[:,0,0,2] = 0

            if cfg.gt_2d:
                predicted_3d_pos[...,:2] = batch_input[...,:2]
            results_all.append(predicted_3d_pos.cpu().numpy())
    results_all = np.concatenate(results_all)
    results_all = datareader.denormalize(results_all)
    _, split_id_test = datareader.get_split_id()
    actions = np.array(datareader.dt_dataset['test']['action'])
    factors = np.array(datareader.dt_dataset['test']['2.5d_factor'])
    gts = np.array(datareader.dt_dataset['test']['joints_2.5d_image'])
    sources = np.array(datareader.dt_dataset['test']['source'])

    num_test_frames = len(actions)
    frames = np.array(range(num_test_frames))
    action_clips = actions[split_id_test]
    factor_clips = factors[split_id_test]
    source_clips = sources[split_id_test]
    frame_clips = frames[split_id_test]
    gt_clips = gts[split_id_test]
    assert len(results_all)==len(action_clips)
    
    e1_all = np.zeros(num_test_frames)
    e2_all = np.zeros(num_test_frames)
    oc = np.zeros(num_test_frames)
    results = {}
    results_procrustes = {}
    action_names = sorted(set(datareader.dt_dataset['test']['action']))
    for action in action_names:
        results[action] = []
        results_procrustes[action] = []
    block_list = ['s_09_act_05_subact_02', 
                  's_09_act_10_subact_02', 
                  's_09_act_13_subact_01']
    for idx in range(len(action_clips)):
        source = source_clips[idx][0][:-6]
        if source in block_list:
            continue
        frame_list = frame_clips[idx]
        action = action_clips[idx][0]
        factor = factor_clips[idx][:,None,None]
        gt = gt_clips[idx]
        pred = results_all[idx]
        pred *= factor
        
        # Root-relative Errors
        pred = pred - pred[:,0:1,:]
        gt = gt - gt[:,0:1,:]
        err1 = mpjpe(pred, gt)
        err2 = p_mpjpe(pred, gt)
        e1_all[frame_list] += err1
        e2_all[frame_list] += err2
        oc[frame_list] += 1
    for idx in range(num_test_frames):
        if e1_all[idx] > 0:
            err1 = e1_all[idx] / oc[idx]
            err2 = e2_all[idx] / oc[idx]
            action = actions[idx]
            results[action].append(err1)
            results_procrustes[action].append(err2)
    final_result = []
    final_result_procrustes = []
    summary_table = prettytable.PrettyTable()
    summary_table.field_names = ['test_name'] + action_names
    for action in action_names:
        final_result.append(np.mean(results[action]))
        final_result_procrustes.append(np.mean(results_procrustes[action]))
    summary_table.add_row(['P1'] + final_result)
    summary_table.add_row(['P2'] + final_result_procrustes)
    print(summary_table)
    e1 = np.mean(np.array(final_result))
    e2 = np.mean(np.array(final_result_procrustes))
    print('Protocol #1 Error (MPJPE):', e1, 'mm')
    print('Protocol #2 Error (P-MPJPE):', e2, 'mm')
    print('----------')
    return e1, e2, results_all

def train_epoch(cfg, model, train_data_loader, loss, optimzer):
    model.train() # Set model to training mode
    for i, (batch_input, batch_gt) in tqdm(enumerate(train_data_loader)):
        batch_size = len(batch_input)
        if torch.cuda.is_available(): # Initilize data on device
            batch_input = batch_input.cuda()
            batch_gt = batch_gt.cuda()
        
        with torch.no_grad():
            batch_gt[:,:,:,2] = batch_gt[:,:,:,2] - batch_gt[:,0:1,0:1,2] # Place the depth of first frame root to 0
            batch_input = cfg.aug.augment2D(batch_input, mask=cfg.mask, noise=cfg.noise)
    
    predicted_3d_pos = model(batch_input) # (N,T,17,3)

    optimzer.zero_grad()
    # Calculate Loss
    loss_3d_pos = loss_mpjpe(predicted_3d_pos, batch_gt)
    loss_3d_scale = n_mpjpe(predicted_3d_pos, batch_gt)
    loss_3d_velocity = loss_velocity(predicted_3d_pos, batch_gt)
    loss_lv = loss_limb_var(predicted_3d_pos)
    loss_lg = loss_limb_gt(predicted_3d_pos, batch_gt)
    loss_a = loss_angle(predicted_3d_pos, batch_gt)
    loss_av = loss_angle_velocity(predicted_3d_pos, batch_gt)
    loss_total = loss_3d_pos + \
                 cfg.lambda_scale       * loss_3d_scale + \
                 cfg.lambda_3d_velocity * loss_3d_velocity + \
                 cfg.lambda_lv          * loss_lv + \
                 cfg.lambda_lg          * loss_lg + \
                 cfg.lambda_a           * loss_a  + \
                 cfg.lambda_av          * loss_av
    # Update loss state with calculated losses
    loss['3d_pos'].update(loss_3d_pos.item(), batch_size)
    loss['3d_scale'].update(loss_3d_scale.item(), batch_size)
    loss['3d_velocity'].update(loss_3d_velocity.item(), batch_size)
    loss['lv'].update(loss_lv.item(), batch_size)
    loss['lg'].update(loss_lg.item(), batch_size)
    loss['angle'].update(loss_a.item(), batch_size)
    loss['angle_velocity'].update(loss_av.item(), batch_size)
    loss['total'].update(loss_total.item(), batch_size)

    loss_total.backward() # Backprop and update grads
    optimzer.step() # Take one step in optimzer

def train(args, cfg):
    print("Training Config:", cfg)

    try:
        os.makedirs(args.save_path)
    except OSError as e:
        raise RuntimeError('Error creating save directory:', args.save_path)
    
    writer = SummaryWriter(os.path.join(args.save_path, "logs")) # Write to logs directory

    print('Loading Dataset')
    

if __name__ == '__main__':
    args = parse_args()
    set_random_seed(args.seed)
    cfg = get_config(args.config)
