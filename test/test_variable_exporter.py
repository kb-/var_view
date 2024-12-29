import pytest
import numpy as np
import torch
from PIL import Image
from pathlib import Path

from variableExporter import VariableExporter


@pytest.fixture
def exporter():
    return VariableExporter()


@pytest.fixture
def sample_data():
    return {
        "numpy_array": np.random.rand(10, 10),
        "torch_tensor": torch.rand(10, 10),
        "string_var": "Hello, World!",
        "nested_list": [[1, 2], [3, 4]],
        "dict_var": {"key1": 1, "key2": "value"},
    }


@pytest.fixture
def tmp_file(tmp_path):
    return tmp_path / "test_file"


def test_save_as_npy(exporter, tmp_file, sample_data):
    file_path = str(tmp_file) + ".npy"
    exporter.save_as_npy(sample_data["numpy_array"], file_path)

    assert Path(file_path).exists()
    loaded = np.load(file_path)
    assert np.allclose(loaded, sample_data["numpy_array"])


def test_save_as_csv(exporter, tmp_file, sample_data):
    file_path = str(tmp_file) + ".csv"
    exporter.save_as_csv(sample_data["numpy_array"], file_path)

    assert Path(file_path).exists()
    loaded = np.loadtxt(file_path, delimiter=",")
    assert np.allclose(loaded, sample_data["numpy_array"])


def test_save_as_h5(exporter, tmp_file, sample_data):
    file_path = str(tmp_file) + ".h5"
    exporter.save_as_h5(sample_data["numpy_array"], file_path)

    import h5py
    assert Path(file_path).exists()
    with h5py.File(file_path, "r") as f:
        loaded = f["data"][:]
        assert np.allclose(loaded, sample_data["numpy_array"])


def test_save_as_mat(exporter, tmp_file, sample_data):
    file_path = str(tmp_file) + ".mat"
    exporter.save_as_mat(sample_data["numpy_array"], file_path)

    import scipy.io as sio
    assert Path(file_path).exists()
    loaded = sio.loadmat(file_path)["data"]
    assert np.allclose(loaded, sample_data["numpy_array"])


def test_save_as_png(exporter, tmp_file, sample_data):
    from pathlib import Path
    import numpy as np
    from PIL import Image

    file_path = str(tmp_file) + ".png"
    array = sample_data["numpy_array"].astype(np.float32)  # Float32 for testing scaling

    # Save the array as PNG
    exporter.save_as_png(array, file_path)

    # Ensure the file was created
    assert Path(file_path).exists()

    # Load the saved PNG and compare
    with Image.open(file_path) as img:
        loaded = np.array(img)

    # Recreate the expected normalized array
    array_min = array.min()
    array_max = array.max()
    if array_min == array_max:
        expected = np.zeros_like(array, dtype=np.uint8)
        expected[:] = 255 if array_min > 0 else 0
    else:
        expected = ((array - array_min) / (array_max - array_min) * 255).astype(
            np.uint8)

    # Assert exact equality
    assert np.array_equal(loaded,
                          expected), f"Mismatch between saved and loaded PNG data."


def test_save_as_txt(exporter, tmp_file, sample_data):
    file_path = str(tmp_file) + ".txt"
    exporter.save_as_txt(sample_data["string_var"], file_path)

    assert Path(file_path).exists()
    with open(file_path, "r") as f:
        loaded = f.read()
        assert loaded == sample_data["string_var"]


def test_save_as_txt_list(exporter, tmp_file, sample_data):
    file_path = str(tmp_file) + ".txt"
    exporter.save_as_txt(sample_data["nested_list"], file_path)

    assert Path(file_path).exists()
    with open(file_path, "r") as f:
        loaded = eval(f.read())  # Use eval carefully for trusted content
        assert loaded == sample_data["nested_list"]


def test_export_unsupported_type(exporter, tmp_file):
    file_path = str(tmp_file) + ".txt"
    exporter.save_as_txt(set([1, 2, 3]), file_path)

    assert Path(file_path).exists()
    with open(file_path, "r") as f:
        loaded = f.read()
        assert "Exported as string representation" in loaded
        assert "{1, 2, 3}" in loaded  # Check the set's string representation


def test_torch_tensor_as_npy(exporter, tmp_file, sample_data):
    file_path = str(tmp_file) + ".npy"
    exporter.save_as_npy(sample_data["torch_tensor"], file_path)

    assert Path(file_path).exists()
    loaded = np.load(file_path)
    assert np.allclose(loaded, sample_data["torch_tensor"].numpy())
