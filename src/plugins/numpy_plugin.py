# src/plugins/numpy_plugin.py
import numpy as np

from src.variable_viewer import VariableRepresentation


def numpy_formatter(value: np.ndarray):
    try:
        nbytes = value.nbytes
        shape = value.shape
        dtype = value.dtype
        return VariableRepresentation(nbytes=nbytes, shape=shape, dtype=dtype)
    except Exception as e:
        return VariableRepresentation(nbytes=0, extra_info=f"<Error: {e}>")


def register_handlers(register_type_handler):
    register_type_handler(np.ndarray, numpy_formatter)
