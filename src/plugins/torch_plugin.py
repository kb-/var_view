# src/plugins/torch_plugin.py
import torch

from src.variable_viewer import VariableRepresentation


def torch_tensor_formatter(value: torch.Tensor):
    shape = tuple(value.shape)
    dtype = value.dtype
    sample = value.flatten().tolist()[:5]  # First 5 elements
    return VariableRepresentation(
        nbytes=value.element_size() * value.nelement(),
        shape=shape,
        dtype=dtype,
        value_summary=f"{sample}... dtype={dtype}, device={value.device}"
    )


def register_handlers(register_type_handler):
    register_type_handler(torch.Tensor, torch_tensor_formatter)