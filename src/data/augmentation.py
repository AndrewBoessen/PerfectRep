import random
import torch
from src.utils.tools import read_pkl
from src.utils.data import flip_data, crop_scale_3d


class Augmenter2D(object):
    """
        Make 2D augmentations on the fly. PyTorch batch-processing GPU version.
    """

    def __init__(self, args):
        self.d2c_params = read_pkl(args.d2c_params_path)
        self.noise = torch.load(args.noise_path)
        self.mask_ratio = args.mask_ratio
        self.mask_T_ratio = args.mask_T_ratio
        self.num_Kframes = 27
        self.noise_std = 0.002

    def dis2conf(self, dis, a, b, m, s):
        """
        Distance to confidence
        """
        f = a/(dis+a)+b*dis
        shift = torch.randn(*dis.shape)*s + m
        # if torch.cuda.is_available():
        shift = shift.to(dis.device)
        return f + shift

    def add_noise(self, motion_2d):
        """
        Add gaussian and uniform noise to the joints for a given sequence of 2d motion clips

        Args:
            motion_2d: motion data (N,T,J,C) (N,243,17,3)
        """
        a, b, m, s = self.d2c_params["a"], self.d2c_params["b"], self.d2c_params["m"], self.d2c_params["s"]
        if "uniform_range" in self.noise.keys():
            uniform_range = self.noise["uniform_range"]
        else:
            uniform_range = 0.06
        # Remove confidence from motion if present
        motion_2d = motion_2d[:, :, :, :2]
        batch_size = motion_2d.shape[0]  # Number of clips in motion data
        num_frames = motion_2d.shape[1]  # Number of frames
        num_joints = motion_2d.shape[2]  # Number of joints in data
        mean = self.noise['mean'].float()
        std = self.noise['std'].float()
        weight = self.noise['weight'][:, None].float()
        # Randomly select joints to add noise to
        sel = torch.rand((batch_size, self.num_Kframes, num_joints, 1))
        gaussian_sample = (torch.randn(
            batch_size, self.num_Kframes, num_joints, 2) * std + mean)  # Gaussian noise
        uniform_sample = (torch.rand(
            (batch_size, self.num_Kframes, num_joints, 2))-0.5) * uniform_range
        noise_mean = 0
        delta_noise = torch.randn(
            num_frames, num_joints, 2) * self.noise_std + noise_mean
        # if torch.cuda.is_available():
        mean = mean.to(motion_2d.device)
        std = std.to(motion_2d.device)
        weight = weight.to(motion_2d.device)
        # Apply gaussian noise to motion data
        gaussian_sample = gaussian_sample.to(motion_2d.device)
        # Apply uniform noise to motion data
        uniform_sample = uniform_sample.to(motion_2d.device)
        sel = sel.to(motion_2d.device)
        delta_noise = delta_noise.to(motion_2d.device)

        delta = gaussian_sample*(sel < weight) + uniform_sample*(sel >= weight)
        delta_expand = torch.nn.functional.interpolate(delta.unsqueeze(
            1), [num_frames, num_joints, 2], mode='trilinear', align_corners=True)[:, 0]
        delta_final = delta_expand + delta_noise
        motion_2d = motion_2d + delta_final
        dx = delta_final[:, :, :, 0]
        dy = delta_final[:, :, :, 1]
        dis2 = dx*dx+dy*dy
        dis = torch.sqrt(dis2)
        conf = self.dis2conf(dis, a, b, m, s).clip(0, 1).reshape(
            [batch_size, num_frames, num_joints, -1])
        return torch.cat((motion_2d, conf), dim=3)

    def add_mask(self, x):
        ''' motion_2d: (N,T,17,3)
            Drop at the joint and frame level
        '''
        N, T, J, C = x.shape
        mask = torch.rand(N, T, J, 1, dtype=x.dtype,
                          device=x.device) > self.mask_ratio  # Drop joints
        mask_T = torch.rand(1, T, 1, 1, dtype=x.dtype,
                            device=x.device) > self.mask_T_ratio  # Drop frames
        x = x * mask * mask_T  # Matrix mult
        return x

    def augment2D(self, motion_2d, mask=False, noise=False):
        """
        Augment a given motion sequence of 2D joints by masking and/or adding noise

        Args:
            motion_2d: Set of motion clips (N, T, J, C) (N, 243, 17, 3)
            mask: Apply masking. optional defaults to False
            noise: Apply noise. optional defaults to False
        """
        if noise:
            motion_2d = self.add_noise(motion_2d)
        if mask:
            motion_2d = self.add_mask(motion_2d)
        return motion_2d


class Augmenter3D(object):
    """
        Make 3D augmentations when dataloaders get items. NumPy single motion version.
    """

    def __init__(self, args):
        self.flip = args.flip
        if hasattr(args, "scale_range_pretrain"):
            self.scale_range_pretrain = args.scale_range_pretrain
        else:
            self.scale_range_pretrain = None

    def augment3D(self, motion_3d):
        """
        Augment a single 3d motion clip
        """
        # motion_3d (N, J, C) (243, 17, 3)
        if self.scale_range_pretrain:
            # Normalize 3D to a given range of vals e.g. [1, 1]
            motion_3d = crop_scale_3d(motion_3d, self.scale_range_pretrain)
        if self.flip and random.random() > 0.5:
            # Flip orientation of joints in motion sequence
            motion_3d = flip_data(motion_3d)
        return motion_3d
