import pytest
import torch
from unittest.mock import patch, MagicMock
from src.data.augmentation import Augmenter2D

class Args:
    d2c_params_path = 'lib/params/d2c_params.pkl'
    noise_path = 'lib/params/sythentic_noise.pth'
    mask_ratio = 0.1
    mask_T_ratio = 0.2

@pytest.fixture
def args():
    return Args()

@pytest.fixture
def augmenter(args):
    with patch('lib.utils.tools.read_pkl') as mock_read_pkl, \
         patch('torch.load') as mock_torch_load:
        
        # Mocking the read_pkl function and torch.load
        mock_read_pkl.return_value = {"a": 0.5, "b": 0.2, "m": 0.1, "s": 0.3}
        mock_torch_load.return_value = {
            "mean": torch.tensor([0.0, 0.0]),
            "std": torch.tensor([1.0, 1.0]),
            "weight": torch.tensor([0.5] * 17),
            "uniform_range": 0.06
        }
        
        return Augmenter2D(args)

def test_add_noise(augmenter):
    motion_2d = torch.randn(5, 243, 17, 3)  # (N, T, J, C)
    result = augmenter.add_noise(motion_2d)
    
    assert result.shape == motion_2d.shape
    assert torch.all(result[:, :, :, :2] != motion_2d[:, :, :, :2]).item()

def test_add_mask(augmenter):
    motion_2d = torch.randn(5, 243, 17, 3)  # (N, T, J, C)
    result = augmenter.add_mask(motion_2d)
    
    assert result.shape == motion_2d.shape
    mask_applied = torch.sum(result == 0) > 0
    assert mask_applied.item()  # Ensure some masking is applied

def test_augment2D_no_mask_no_noise(augmenter):
    motion_2d = torch.randn(5, 243, 17, 3)
    result = augmenter.augment2D(motion_2d, mask=False, noise=False)
    
    assert torch.equal(result, motion_2d)

def test_augment2D_mask(augmenter):
    motion_2d = torch.randn(5, 243, 17, 3)
    result = augmenter.augment2D(motion_2d, mask=True, noise=False)
    
    assert result.shape == motion_2d.shape
    mask_applied = torch.sum(result == 0) > 0
    assert mask_applied.item()  # Ensure some masking is applied

def test_augment2D_noise(augmenter):
    motion_2d = torch.randn(5, 243, 17, 3)
    result = augmenter.augment2D(motion_2d, mask=False, noise=True)
    
    assert result.shape == motion_2d.shape
    assert torch.all(result[:, :, :, :2] != motion_2d[:, :, :, :2]).item()

def test_augment2D_mask_and_noise(augmenter):
    motion_2d = torch.randn(5, 243, 17, 3)
    result = augmenter.augment2D(motion_2d, mask=True, noise=True)
    
    assert result.shape == motion_2d.shape
    assert torch.all(result[:, :, :, :2] != motion_2d[:, :, :, :2]).item()
    mask_applied = torch.sum(result == 0) > 0
    assert mask_applied.item()  # Ensure some masking is applied