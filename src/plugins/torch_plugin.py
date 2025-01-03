# torch_plugin.py
import torch


def torch_tensor_formatter(value: torch.tensor):
    shape = tuple(value.shape)
    dtype = value.dtype
    sample = value.flatten().tolist()[:5]  # First 5 elements
    return f"Torch Tensor{shape} ({dtype}): {sample}..."


def register_handlers(register_type_handler):
    register_type_handler(torch.Tensor, torch_tensor_formatter)
