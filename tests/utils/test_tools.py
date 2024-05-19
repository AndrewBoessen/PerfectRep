import os
import pickle
import pytest
from src.utils.tools import ensure_dir, read_pkl


@pytest.fixture
def tmp_dir(tmpdir):
    # Fixture to create a temporary directory for testing
    return str(tmpdir.mkdir("test_dir"))


def test_ensure_dir(tmp_dir):
    # Test if ensure_dir creates a directory if it doesn't exist
    test_path = os.path.join(tmp_dir, "new_dir")
    ensure_dir(test_path)
    assert os.path.exists(test_path)
    assert os.path.isdir(test_path)


def test_read_pkl(tmp_dir):
    # Test reading a pickle file
    test_data = {"key": "value"}
    test_file = os.path.join(tmp_dir, "test.pkl")

    # Write test data to a pickle file
    with open(test_file, "wb") as f:
        pickle.dump(test_data, f)

    # Read the pickle file and compare the content
    content = read_pkl(test_file)
    assert content == test_data


def test_read_pkl_nonexistent_file(tmp_dir):
    # Test reading a non-existent pickle file
    non_existent_file = os.path.join(tmp_dir, "non_existent.pkl")
    with pytest.raises(FileNotFoundError):
        read_pkl(non_existent_file)
