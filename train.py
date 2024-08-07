import os
import numpy as np
import argparse
import errno
from functools import partial
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
from src.utils.training import *
from src.utils.data import flip_data
from src.data.dataset_motion_3d import MotionDataset3D
from src.data.augmentation import Augmenter2D
from src.data.datareader_fit3d import DataReaderFit3D
from src.model.loss import *
from src.model.DSTformer import DSTformer

def parse_args():
    parser = argparse.ArgumentParser(
        prog='PerfectRep Train',
        description='A script for pretraining the 3D pose estimation model. This trains the model on the fit3d dataset'
    )
    # Add Arguments for Data and Hyperparameters
    parser.add_argument('--config', default='train_config.yaml', type=str, metavar='FILENAME', help='config file')
    parser.add_argument('-s', '--save_path', default='save', type=str, metavar='PATH', help='path to store logs and checkpoint saves')
    parser.add_argument('-d', '--data_path', default='data/motion3d', type=str, metavar='PATH', help='path to training data directory')
    parser.add_argument('-r', '--random_seed', default=0, type=int, help='seed used to generate random numbers')
    parser.add_argument('-c', '--checkpoint', default=None, type=str, metavar='FILENAME', help='filename of checkpoint binary to load (e.g. best_epoch.bin file)')
    parser.add_argument('-v', '--evaluate', default=False, action='store_true', help='Don\'t train, only evaluate accuracy of given checkpoint')
    parser.add_argument('-l', '--selection', default='best_epoch.bin', type=str, metavar='FILENAME', help='checkpoint to finetune (file name)')
    parser.add_argument('-b', '--batch_size', default=3, type=int, help='batch size for training')
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
    print('Saving checkpoint to', chk_path)
    torch.save({
        'epoch': epoch + 1,
        'lr': lr,
        'optimizer': optimizer.state_dict(),
        'model': model.state_dict(),
        'min_loss' : min_loss
    }, chk_path)

def evaluate(cfg, model, test_loader, datareader):
    print("Evaluating Model")
    results_all = []
    gts = []
    model.eval() # Put model into evaluation mode

    with torch.no_grad():
        for input, gt in tqdm(test_loader):
            N, T = gt.shape[:2]
            if torch.cuda.is_available():
                input = input.cuda()
            pred_3d = model(input)

            gts.append(gt.cpu().numpy())
            results_all.append(pred_3d.cpu().numpy()) # Convert to np array and append to results
    results_all = np.concatenate(results_all)
    results_all = datareader.denormalize(results_all)
    gts = np.concatenate(gts)
    gts = datareader.denormalize(gts)
    gt_clips = gts

    _, split_id_test = datareader.get_split_id()

    # gts = np.array(datareader.dt_dataset['test']['3d_joint_labels'])
    sources = np.array(datareader.dt_dataset['test']['source'])
    actions = np.array(datareader.dt_dataset['test']['actions'])

    num_test_frames = len(sources)
    frames = np.array(range(num_test_frames))

    action_clips = actions[split_id_test]
    source_clips = sources[split_id_test]
    frame_clips = frames[split_id_test]
    assert len(results_all) == len(action_clips)

    e1_all = np.zeros(num_test_frames)
    e2_all = np.zeros(num_test_frames)

    oc = np.zeros(num_test_frames)
    results = {}
    results_procrustes = {}
    action_names = sorted(set(datareader.dt_dataset['test']['actions']))
    for action in action_names:
        results[action] = []
        results_procrustes[action] = []

    for i in range(len(action_clips)):
        frame_list = frame_clips[i]
        
        action = action_clips[i][0]
        gt = gt_clips[i]
        pred = results_all[i]

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

    return e1, e2

def train_epoch(cfg, model, train_data_loader, loss, optimzer):
    model.train() # Set model to training mode
    for i, (batch_input, batch_gt) in tqdm(enumerate(train_data_loader)):

        batch_size = len(batch_input)
        if torch.cuda.is_available(): # Initilize data on device
            batch_input = batch_input.cuda()
            batch_gt = batch_gt.cuda()
        
        with torch.no_grad():
            batch_gt[:,:,:,2] = batch_gt[:,:,:,2] - batch_gt[:,0:1,0:1,2] # Place the depth of first frame root to 0
            if cfg.noise or cfg.mask:
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
        os.makedirs(args.save_path, exist_ok=True)
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

    train_dataset = MotionDataset3D(cfg, cfg.subset_list, 'train')
    test_dataset = MotionDataset3D(cfg, cfg.subset_list, 'test')
    train_loader_3d = DataLoader(train_dataset, **train_loader_params)
    test_loader = DataLoader(test_dataset, **test_loader_params)

    datareader_h36m = DataReaderFit3D(n_frames=cfg.clip_len, sample_stride=cfg.sample_stride, data_stride_train=cfg.data_stride, data_stride_test=cfg.clip_len, dt_root = args.data_path, dt_file=cfg.dt_file)
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
    
    if cfg.finetune:
        checkpoint_file = os.path.join(args.save_path, args.selection)
        if os.path.exists(checkpoint_file):
            print('Loading checkpoint to finetune', checkpoint_file)
            checkpoint = torch.load(checkpoint_file, map_location=lambda storage, loc: storage)
            model_backbone.load_state_dict(checkpoint['model'], strict=True)
        else:
            warnings.warn("Checkpoint file does not exist: %s" % checkpoint_file)
            checkpoint = None   
    else:
        if args.checkpoint and args.checkpoint != None:
            checkpoint_file = os.path.join(args.save_path, args.checkpoint)
            if os.path.exists(checkpoint_file):
                print('Loading Checkpoint', checkpoint_file)
                checkpoint = torch.load(checkpoint_file, map_location=lambda storage, loc: storage)
                model_backbone.load_state_dict(checkpoint['model'], strict=True)
            else:
                warnings.warn("Checkpoint file does not exist: %s" % checkpoint_file)
                checkpoint = None
        else:
            checkpoint = None
    model = model_backbone

    if not args.evaluate: # Not in evaluation mode
        lr = cfg.learning_rate
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=cfg.weight_decay)
        lr_decay = cfg.lr_decay
        st = 0

        print("Training on %s Batches" % len(train_loader_3d))

        # Load checkpoint if given
        if checkpoint:
            st = checkpoint['epoch'] if not cfg.finetune else 0
            if 'optimizer' in checkpoint and checkpoint['optimizer'] != None:
                optimizer.load_state_dict(checkpoint['optimizer'])
            else:
                warnings.warn("Checkpoint does not contain an optimzer. The optimzer will be reset")
            lr = checkpoint['lr'] # Set learning rate from checkpoint
            if 'min_loss' in checkpoint and 'min_loss' != None:
                min_loss = checkpoint['min_loss'] # Upadte min loss from checkpoint

        cfg.mask = (cfg.mask_ratio > 0 and cfg.mask_T_ratio > 0)
        if cfg.mask or cfg.noise:
            cfg.aug = Augmenter2D(cfg) # Data Augmentation: flip and add noise

        for epoch in range(st, args.epochs or cfg.epochs): # Start training
            print("Training Epoch %d" % (epoch + 1))
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
            train_epoch(cfg, model, train_loader_3d, losses, optimizer) 
            elapsed = (time() - start_time) / 60

            e1, e2 = evaluate(cfg, model, test_loader, datareader_h36m) # Evaluate loss after epoch

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
            
            ensure_dir(args.save_path)
            assert os.path.exists(args.save_path), "Error saving checkpoint: %s path does not exist!" % arg.save_path

            # Save checkpoints
            chk_path_latest = os.path.join(args.save_path, 'latest_epoch.bin')
            chk_path_best = os.path.join(args.save_path, 'best_epoch.bin'.format(epoch))

            save_checkpoint(chk_path_latest, epoch, lr, optimizer, model, min_loss)
            if e1 < min_loss:
                min_loss = e1
                save_checkpoint(chk_path_best, epoch, lr, optimizer, model, min_loss)
    
    if args.evaluate:
        e1, e2 = evaluate(cfg, model, test_loader, datareader_h36m)

if __name__ == '__main__':
    args = parse_args()
    set_random_seed(args.random_seed)
    cfg = get_config(args.config)
    train(args, cfg)
