import copy
import numpy as np
import pytest
from lib.utils.data import crop_scale, crop_scale_3d, flip_data, resample


@pytest.fixture
def mock_motion_data_2d():
    # Mock 2D motion data with shape (1, 17, 3)
    return np.random.uniform(-1, 1, size=(1, 243, 17, 3))


@pytest.fixture
def mock_motion_data_3d():
    # Mock 3D motion data with shape (T, 17, 3)
    return np.random.uniform(-1, 1, size=(1, 17, 3))


def test_crop_scale_with_zero_scale(mock_motion_data_2d):
    # Test crop_scale with zero scale
    motion = np.zeros_like(mock_motion_data_2d)
    result = crop_scale(motion)
    assert np.all(result == 0)


def test_crop_scale_with_nonzero_scale(mock_motion_data_2d):
    # Test crop_scale with non-zero scale
    result = crop_scale(mock_motion_data_2d)
    assert result.shape == mock_motion_data_2d.shape
    assert np.all(result >= -1) and np.all(result <= 1)


def test_crop_scale_3d_with_zero_scale(mock_motion_data_3d):
    # Test crop_scale_3d with zero scale
    motion = np.zeros_like(mock_motion_data_3d)
    result = crop_scale_3d(motion)
    assert np.all(result == 0)


def test_crop_scale_3d_with_nonzero_scale(mock_motion_data_3d):
    # Test crop_scale_3d with non-zero scale
    result = crop_scale_3d(mock_motion_data_3d)
    assert result.shape == mock_motion_data_3d.shape
    assert result[0, 0, 2] == -1  # Z realtive to first frame


def test_flip_data(mock_motion_data_2d):
    data_copy = copy.deepcopy(mock_motion_data_2d)
    flipped_data = flip_data(mock_motion_data_2d)

    # Check if left and right joints are swapped
    left_joints = [4, 5, 6, 11, 12, 13]
    right_joints = [1, 2, 3, 14, 15, 16]
    assert np.all(np.abs(flipped_data[..., left_joints, :]) == np.abs(
        data_copy[..., right_joints, :]))
    assert np.all(np.abs(flipped_data[..., right_joints, :]) == np.abs(
        data_copy[..., left_joints, :]))


def test_flip_data_no_change():
    data = np.zeros((1, 10, 17, 3))
    flipped_data = flip_data(data)
    assert np.all(flipped_data == data)


def test_resample_replay():
    ori_len = 10
    target_len = 5
    replayed_indices = resample(ori_len, target_len, replay=True)
    assert len(replayed_indices) == target_len
    assert all(idx < ori_len for idx in replayed_indices)


def test_resample_no_replay():
    ori_len = 10
    target_len = 5
    resampled_indices = resample(ori_len, target_len, replay=False)
    assert len(resampled_indices) == target_len
    assert all(idx < ori_len for idx in resampled_indices)


def test_resample_with_randomness():
    ori_len = 10
    target_len = 5
    resampled_indices = resample(
        ori_len, target_len, replay=False, randomness=True)
    assert len(resampled_indices) == target_len
    assert all(idx < ori_len for idx in resampled_indices)


def test_resample_no_randomness():
    ori_len = 10
    target_len = 5
    resampled_indices = resample(
        ori_len, target_len, replay=False, randomness=False)
    assert len(resampled_indices) == target_len
    assert all(idx < ori_len for idx in resampled_indices)
