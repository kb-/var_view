# var_view/built_in_plugins/torch_plugin.py
from __future__ import annotations
from typing import TYPE_CHECKING

from var_view.plugin_base import PluginBase
from var_view.variable_viewer import VariableRepresentation

if TYPE_CHECKING:
    import torch  # only imported by static checkers

class TorchPlugin(PluginBase):
    """
    Plugin to handle torch.Tensor objects.
    """

    def register_handlers(self, register_type_handler) -> None:
        try:
            import torch
        except ModuleNotFoundError:
            # torch isn’t installed → skip this plugin
            return

        # now that torch is present, register the formatter
        register_type_handler(torch.Tensor, self.torch_tensor_formatter)

    def torch_tensor_formatter(self, value: torch.Tensor) -> VariableRepresentation:
        """
        Format a torch.Tensor for display in the variable viewer.
        """
        shape = tuple(value.shape)
        dtype = str(value.dtype)
        sample = value.flatten().tolist()[:5]  # First 5 elements
        return VariableRepresentation(
            nbytes=value.element_size() * value.nelement(),
            shape=shape,
            dtype=dtype,
            value_summary=f"{sample}... dtype={dtype}, device={value.device}"
        )
