# src/plugins/torch_plugin.py
import torch

from src.variable_viewer import VariableRepresentation


def torch_formatter(value: torch.Tensor):
    try:
        nbytes = value.element_size() * value.numel()
        shape = tuple(value.shape)
        dtype = value.dtype
        return VariableRepresentation(nbytes=nbytes, shape=shape, dtype=dtype)
    except Exception as e:
        return VariableRepresentation(nbytes=0, extra_info=f"<Error: {e}>")


def register_handlers(register_type_handler):
    register_type_handler(torch.Tensor, torch_formatter)
