import pytest
import numpy as np
import torch
import scipy.io as sio
from pathlib import Path

from src.app.variableExporter import VariableExporter

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
    exporter.save_as_h5("numpy_array", sample_data["numpy_array"], file_path)

    assert Path(file_path).exists()
    import h5py
    with h5py.File(file_path, "r") as f:
        loaded = f["numpy_array"][:]
        assert np.allclose(loaded, sample_data["numpy_array"])

def test_save_as_mat_all_types(exporter, tmp_file, sample_data):
    for variable_name, value in sample_data.items():
        file_path = tmp_file / f"{variable_name}.mat"
        file_path.parent.mkdir(parents=True, exist_ok=True)  # Ensure directory exists

        # Export the variable
        exporter.save_as_mat(variable_name, value, str(file_path))

        # Validate the exported .mat file
        assert file_path.exists(), f"File {file_path} was not created."

        # Load the .mat file and check the contents
        loaded = sio.loadmat(file_path)

        # Check that the variable name is preserved
        assert variable_name in loaded, f"Variable '{variable_name}' not found in .mat file."

        # Validate the data
        if isinstance(value, np.ndarray):
            assert np.allclose(loaded[variable_name], value)
        elif torch.is_tensor(value):
            assert np.allclose(loaded[variable_name], value.cpu().numpy())
        elif isinstance(value, (list, tuple)):
            assert loaded[variable_name].tolist() == list(value)
        elif isinstance(value, str):
            assert loaded[variable_name] == value
        elif isinstance(value, (int, float)):
            assert loaded[variable_name].item() == value


def test_save_as_png(exporter, tmp_file, sample_data):
    file_path = str(tmp_file) + ".png"
    array = sample_data["numpy_array"].astype(np.float32)

    exporter.save_as_png(array, file_path)
    assert Path(file_path).exists()

    from PIL import Image
    with Image.open(file_path) as img:
        loaded = np.array(img)
        expected = ((array - array.min()) / (array.max() - array.min()) * 255).astype(np.uint8)
        assert np.array_equal(loaded, expected)

def test_save_as_txt(exporter, tmp_file, sample_data):
    file_path = str(tmp_file) + ".txt"
    exporter.save_as_txt("string_var", sample_data["string_var"], file_path)

    assert Path(file_path).exists()
    with open(file_path, "r") as f:
        loaded = f.read()
        assert loaded == sample_data["string_var"]

def test_save_as_txt_list(exporter, tmp_file, sample_data):
    file_path = str(tmp_file) + ".txt"
    exporter.save_as_txt("nested_list", sample_data["nested_list"], file_path)

    assert Path(file_path).exists()
    with open(file_path, "r") as f:
        loaded = eval(f.read())
        assert loaded == sample_data["nested_list"]


def test_export_unsupported_type(exporter, tmp_file):
    file_path = str(tmp_file) + ".txt"
    variable_name = "unsupported_type"
    unsupported_value = set([1, 2, 3])

    # Export the unsupported type
    exporter.save_as_txt(variable_name, unsupported_value, file_path)

    assert Path(file_path).exists()
    with open(file_path, "r") as f:
        loaded = f.read()

    # Check the exported content
    expected_header = f"# Exported variable: {variable_name}"
    expected_content = str(unsupported_value)
    assert loaded.startswith(
        expected_header), f"Expected header '{expected_header}' not found in exported content."
    assert expected_content in loaded, f"Expected content '{expected_content}' not found in exported content."


def test_torch_tensor_as_npy(exporter, tmp_file, sample_data):
    file_path = str(tmp_file) + ".npy"
    exporter.save_as_npy(sample_data["torch_tensor"], file_path)

    assert Path(file_path).exists()
    loaded = np.load(file_path)
    assert np.allclose(loaded, sample_data["torch_tensor"].numpy())
