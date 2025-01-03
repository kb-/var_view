# numpy_plugin.py
import numpy as np


def numpy_variable_formatter(value: np.ndarray):
    shape = value.shape
    dtype = value.dtype
    sample = value.flatten()[:5].tolist()  # First 5 elements
    return f"NumPy ndarray{shape} ({dtype}): {sample}..."


def register_handlers(register_type_handler):
    register_type_handler(np.ndarray, numpy_variable_formatter)
