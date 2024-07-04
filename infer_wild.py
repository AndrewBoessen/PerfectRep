import os
import numpy as np
import argparse
from tqdm import tqdm
import imageio
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.utils.tools import *
from src.utils.learning import *
from src.utils.utils_data import flip_data
from src.utils.vismo import render_and_save
from src.data.dataset_wild import WildDetDataset

def parse_args():
    parser = argparse.ArgumentParser(
        prog='PerfectRep Inference',
        description='Perform in-the-wild inference on a single image or a whole video'
    )
    parser.add_argument('--config', default='train_config.yaml', type=str, metavar='FILENAME', help='config file')
    parser.add_argument('-c', '--checkpoint', type=PATH, help='checkpoint file location')
    parser.add_argument('-v', '--video', type=str, help='video path')
    parser.add_argument('-i', '--image', type=str, help='image path')
    parser.add_argument('-j', '--json_path', type=str, help='alphapose detection result json path')
    parser.add_argument('-o', '--out_path', type=str, help='output path')
    parser.add_argument('--focus', type=int, default=None, help='target person id')
    parser.add_argument('--pixel', action='store_true', help='align with pixle coordinates')

args = parse_args()
cfg = get_config(args.config)

model_backbone =  DSTformer(dim_in=3, dim_out=3, dim_feat=cfg.dim_feat, dim_rep=cfg.dim_rep, 
                            depth=cfg.depth, num_heads=cfg.num_heads, mlp_ratio=cfg.mlp_ratio, norm_layer=partial(nn.LayerNorm, eps=1e-6), 
                            maxlen=cfg.maxlen, num_joints=cfg.num_joints)
if torch.cuda.is_available():
    model_backbone = nn.DataParallel(model_backbone)
    model_backbone = model_backbone.cuda()

print('Loading checkpoint', args.checkpoint)
checkpoint = torch.load(args.checkpoint, map_location=lambda storage, loc: storage)
model_backbone.load_state_dict(checkpoint['model_pos'], strict=True)
model_pos = model_backbone
model_pos.eval()
testloader_params = {
    'batch_size': 1,
    'shuffle': False,
    'num_workers': 8,
    'pin_memory': True,
    'prefetch_factor': 4,
    'persistent_workers': True,
    'drop_last': False
}

vid = imageio.get_reader(args.video,  'ffmpeg')
fps_in = vid.get_meta_data()['fps']
vid_size = vid.get_meta_data()['size']
os.makedirs(args.out_path, exist_ok=True)

if args.pixel:
    # Keep relative scale with pixel coornidates
    wild_dataset = WildDetDataset(args.json_path, clip_len=243, vid_size=vid_size, scale_range=None, focus=args.focus)
else:
    # Scale to [-1,1]
    wild_dataset = WildDetDataset(args.json_path, clip_len=243, scale_range=[1,1], focus=args.focus)

test_loader = DataLoader(wild_dataset, **testloader_params)

results_all = []
with torch.no_grad():
    for batch_input in tqdm(test_loader):
        N, T = batch_input.shape[:2]
        if torch.cuda.is_available():
            batch_input = batch_input.cuda()
        if args.no_conf:
            batch_input = batch_input[:, :, :, :2]
        if args.flip:    
            batch_input_flip = flip_data(batch_input)
            predicted_3d_pos_1 = model_pos(batch_input)
            predicted_3d_pos_flip = model_pos(batch_input_flip)
            predicted_3d_pos_2 = flip_data(predicted_3d_pos_flip) # Flip back
            predicted_3d_pos = (predicted_3d_pos_1 + predicted_3d_pos_2) / 2.0
        else:
            predicted_3d_pos = model_pos(batch_input)
        if args.rootrel:
            predicted_3d_pos[:,:,0,:]=0                    # [N,T,17,3]
        else:
            predicted_3d_pos[:,0,0,2]=0
            pass
        if args.gt_2d:
            predicted_3d_pos[...,:2] = batch_input[...,:2]
        results_all.append(predicted_3d_pos.cpu().numpy())

results_all = np.hstack(results_all)
results_all = np.concatenate(results_all)
render_and_save(results_all, '%s/X3D.mp4' % (args.out_path), keep_imgs=False, fps=fps_in)
if args.pixel:
    # Convert to pixel coordinates
    results_all = results_all * (min(vid_size) / 2.0)
    results_all[:,:,:2] = results_all[:,:,:2] + np.array(vid_size) / 2.0
np.save('%s/X3D.npy' % (args.out_path), results_all)