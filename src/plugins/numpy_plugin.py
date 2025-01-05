# src/plugins/numpy_plugin.py
import numpy as np

from src.variable_viewer import VariableRepresentation


def numpy_variable_formatter(value: np.ndarray):
    shape = value.shape
    dtype = value.dtype
    sample = value.flatten()[:5].tolist()  # First 5 elements
    return VariableRepresentation(
        nbytes=value.nbytes,
        shape=shape,
        dtype=dtype,
        value_summary=f"sample={sample}... dtype={dtype}"
    )


def register_handlers(register_type_handler):
    register_type_handler(np.ndarray, numpy_variable_formatter)