# var_view/built_in_plugins/torch_plugin.py

import torch

from var_view.plugin_base import PluginBase
from var_view.variable_viewer import VariableRepresentation


class TorchPlugin(PluginBase):
    """
    Plugin to handle torch.Tensor objects.
    """

    def register_handlers(self, register_type_handler):
        register_type_handler(torch.Tensor, self.torch_tensor_formatter)

    def torch_tensor_formatter(self, value: torch.Tensor) -> VariableRepresentation:
        shape = tuple(value.shape)
        dtype = str(value.dtype)
        sample = value.flatten().tolist()[:5]  # First 5 elements
        return VariableRepresentation(
            nbytes=value.element_size() * value.nelement(),
            shape=shape,
            dtype=dtype,
            value_summary=f"{sample}... dtype={dtype}, device={value.device}"
        )
