import pytest
import torch
from unittest.mock import Mock, patch
from lib.data.augmentation import Augmenter2D, Augmenter3D

# Sample data for testing
motion_2d = torch.randn(2, 243, 17, 3)  # (N, T, J, C)
motion_3d = torch.randn(243, 17, 3)     # (T, J, C)

# Mock arguments
class Args:
    d2c_params_path = 'mock_d2c_params.pkl'
    noise_path = 'mock_noise.pt'
    mask_ratio = 0.1
    mask_T_ratio = 0.1
    flip = True
    scale_range_pretrain = [0.8, 1.2]

args = Args()

@pytest.fixture
def setup_augmenter2d():
    with patch('lib.utils.tools.read_pkl') as mock_read_pkl, \
         patch('torch.load') as mock_torch_load:
        mock_read_pkl.return_value = {"a": 1.0, "b": 0.5, "m": 0.1, "s": 0.02}
        mock_torch_load.return_value = {
            "mean": torch.tensor([0.0, 0.0]),
            "std": torch.tensor([1.0, 1.0]),
            "weight": torch.tensor([0.5]),
            "uniform_range": 0.06
        }
        augmenter = Augmenter2D(args)
    return augmenter

@pytest.fixture
def setup_augmenter3d():
    return Augmenter3D(args)

def test_add_noise(setup_augmenter2d):
    augmenter = setup_augmenter2d
    augmented_motion = augmenter.add_noise(motion_2d)
    assert augmented_motion.shape == (2, 243, 17, 3), "Shape mismatch in add_noise output"
    assert (augmented_motion[:, :, :, :2] != motion_2d[:, :, :, :2]).any(), "Noise not added correctly"

def test_add_mask(setup_augmenter2d):
    augmenter = setup_augmenter2d
    masked_motion = augmenter.add_mask(motion_2d)
    assert masked_motion.shape == (2, 243, 17, 3), "Shape mismatch in add_mask output"
    mask_ratio = (masked_motion == 0).float().mean()
    assert mask_ratio <= args.mask_ratio + 0.05, "Mask ratio not within expected range"

def test_augment2D(setup_augmenter2d):
    augmenter = setup_augmenter2d
    augmented_motion = augmenter.augment2D(motion_2d, mask=True, noise=True)
    assert augmented_motion.shape == (2, 243, 17, 3), "Shape mismatch in augment2D output"

def test_augment3D(setup_augmenter3d):
    augmenter = setup_augmenter3d
    augmented_motion = augmenter.augment3D(motion_3d)
    assert augmented_motion.shape == (243, 17, 3), "Shape mismatch in augment3D output"
    if args.flip:
        # Test flip (probabilistic)
        flipped_motion = augmenter.augment3D(motion_3d)
        assert (flipped_motion != motion_3d).any(), "Flip augmentation not applied"

def test_augment3D_scale_range(setup_augmenter3d):
    augmenter = setup_augmenter3d
    augmenter.scale_range_pretrain = [0.8, 1.2]
    scaled_motion = augmenter.augment3D(motion_3d)
    assert scaled_motion.shape == (243, 17, 3), "Shape mismatch in augment3D output with scale range"
