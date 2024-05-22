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
import warnings

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
from src.data.datareader_fit3d import DataReaderFit3D
from src.model.loss import *
from lib.model.DSTformer import DSTformer

def parse_args():
    parser = argparse.ArgumentParser(
        prog='PerfectRep Train',
        description='A script for pretraining the 3D pose estimation model. This trains the model on the h3.6m and optional fit3d dataset'
    )
    # Add Arguments for Data and Hyperparameters
    parser.add_argument('--config', default='train_config.yaml', type=str, metavar='FILENAME', help='config file')
    parser.add_argument('-s', '--save_path', default='save/', type=str, metavar='PATH', help='path to store logs and checkpoint saves')
    parser.add_argument('-d', '--data_path', default='data/', type=str, metavar='PATH', help='path to training data directory')
    parser.add_argument('-c', '--checkpoint', default='', type=str, metavar='FILENAME', help='filename of checkpoint binary to load (e.g. model.pt file)')
    parser.add_argument('-v', '--evaluate', default=False, action='store_true', help='evaluate accuracy of given checkpoint')
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
    train_loader_params = {
        'batch_size': args.batch_size,
        'shuffle': True,
        'num_workers': 12,
        'pin_memory': True,
        'prefetch_factor': 4,
        'persistent_workers': True
    }

    test_loader_params = {
        'batch_size': args.batch_size,
        'shuffle': False,
        'num_workers': 12,
        'pin_memory': True,
        'prefetch_factor': 4,
        'persistent_workers': True
    }

    train_dataset = MotionDataset3D(args, args.subset_list, 'train')
    test_dataset = MotionDataset3D(args, args.subset_list, 'test')
    train_loader_3d = DataLoader(train_dataset, **trainloader_params)
    test_loader = DataLoader(test_dataset, **testloader_params)

    datareader_h36m = DataReaderFit3D(n_frames=cfg.clip_len, sample_stride=cfg.sample_stride, data_stride_train=cfg.data_stride, data_stride_test=cfg.clip_len, dt_root = args.data_path, dt_file=cfg.h36m_file)
    min_loss = 100000
    model_backbone =  DSTformer(dim_in=3, dim_out=3, dim_feat=cfg.dim_feat, dim_rep=cfg.dim_rep, 
                                depth=cfg.depth, num_heads=cfg.num_heads, mlp_ratio=cfg.mlp_ratio, norm_layer=partial(nn.LayerNorm, eps=1e-6), 
                                maxlen=cfg.maxlen, num_joints=cfg.num_joints)
    model_params = 0
    for parameter in model_backbone.parameters(): # Get number of model parameters
        model_params = model_params + parameter.numel()
    print('Model Parameter Count:', model_params)

    if torch.cuda.is_available():
        model_backbone = nn.DataParallel(model_backbone)
        model_backbone = model_backbone.cuda()
    
    checkpoint_file = os.path.join(args.save_path, args.checkpoint)
    if os.path.exists(checkpoint_file):
        print('Loading Checkpoint', checkpoint_file)
        checkpoint = torch.load(checkpoint_file, map_location=lambda storage, loc: storage)
        model_backbone.load_state_dict(checkpoint['model_pos'], strict=True)
    model = model_backbone

    if not args.evaluate: # Not in evaluation mode
        lr = cfg.learning_rate
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model_pos.parameters()), lr=lr, weight_decay=cfg.weight_decay)
        lr_decay = cfg.lr_decay
        st = 0

        print("Training on %s Batches" % len(train_loader_3d))

        # Load checkpoint if given
        if checkpoint:
            st = checkpoint['epoch']
            if 'optimizer' in checkpoint and checkpoint['optimzer'] != None:
                optimizer.load_state_dict(checkpoint['optimizer'])
            else:
                warnings.warn("Checkpoint does not contain an optimzer. The optimzer will be reset")
        lr = checkpoint['lr']
        if 'min_loss' in checkpoint and 'min_loss' != None:
            min_loss = checkpoint['min_loss']
        
        for epoch in range(st, cfg.epochs): # Start training
            print("Training Epoch %d" % epoch)
            start_time = time()
            # Reset losses for current epoch
            losses = {}
            losses['3d_pos'] = AverageMeter()
            losses['3d_scale'] = AverageMeter()
            losses['2d_proj'] = AverageMeter()
            losses['lg'] = AverageMeter()
            losses['lv'] = AverageMeter()
            losses['total'] = AverageMeter()
            losses['3d_velocity'] = AverageMeter()
            losses['angle'] = AverageMeter()
            losses['angle_velocity'] = AverageMeter()
            N = 0
            # Train in 3D data
            train_epoch(args, model_pos, train_loader_3d, losses, optimizer) 
            elapsed = (time() - start_time) / 60

            e1, e2, results = evaluate(cfg, model, test_loader, datareader_h36m) # Evaluate loss after epoch

            print('[%d] time %.2f lr %f 3d_train %f e1 %f e2 %f' % (
                epoch + 1,
                elapsed,
                lr,
                losses['3d_pos'].avg,
                e1, e2))
            # Write to training log
            writer.add_scalar('Error P1', e1, epoch + 1)
            writer.add_scalar('Error P2', e2, epoch + 1)
            writer.add_scalar('loss_3d_pos', losses['3d_pos'].avg, epoch + 1)
            writer.add_scalar('loss_2d_proj', losses['2d_proj'].avg, epoch + 1)
            writer.add_scalar('loss_3d_scale', losses['3d_scale'].avg, epoch + 1)
            writer.add_scalar('loss_3d_velocity', losses['3d_velocity'].avg, epoch + 1)
            writer.add_scalar('loss_lv', losses['lv'].avg, epoch + 1)
            writer.add_scalar('loss_lg', losses['lg'].avg, epoch + 1)
            writer.add_scalar('loss_a', losses['angle'].avg, epoch + 1)
            writer.add_scalar('loss_av', losses['angle_velocity'].avg, epoch + 1)
            writer.add_scalar('loss_total', losses['total'].avg, epoch + 1)
                
            # Decay learning rate exponentially
            lr *= lr_decay
            for param_group in optimizer.param_groups:
                param_group['lr'] *= lr_decay

            # Save checkpoints
            chk_path_latest = os.path.join(opts.checkpoint, 'latest_epoch.bin')
            chk_path_best = os.path.join(opts.checkpoint, 'best_epoch.bin'.format(epoch))

            save_checkpoint(chk_path_latest, epoch, lr, optimizer, model, min_loss)
            if e1 < min_loss:
                min_loss = e1
                save_checkpoint(chk_path_best, epoch, lr, optimizer, model, min_loss)
    
    if args.evaluate:
        e1, e2, results = evaluate(cfg, model, test_loader, datareader_h36m)

if __name__ == '__main__':
    args = parse_args()
    set_random_seed(args.seed)
    cfg = get_config(args.config)
    train(args, cfg)
