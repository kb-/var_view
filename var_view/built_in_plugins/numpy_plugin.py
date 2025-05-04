# var_view/built_in_plugins/numpy_plugin.py
from __future__ import annotations
from typing import TYPE_CHECKING

from var_view.plugin_base import PluginBase
from var_view.variable_viewer import VariableRepresentation

if TYPE_CHECKING:
    import numpy as np  # only for static type checkers

class NumpyArrayPlugin(PluginBase):
    """
    Plugin to handle numpy.ndarray objects.
    """

    def register_handlers(self, register_type_handler) -> None:
        try:
            import numpy as np
        except ModuleNotFoundError:
            # NumPy not installed → skip this plugin
            return

        register_type_handler(np.ndarray, self.numpy_array_formatter)

    def numpy_array_formatter(self, value: np.ndarray) -> VariableRepresentation:
        """
        Format a numpy.ndarray by summarizing its size, shape, dtype, and a small sample.
        """
        shape = value.shape
        dtype = str(value.dtype)
        sample = value.flatten()[:5].tolist()  # First 5 elements
        return VariableRepresentation(
            nbytes=value.nbytes,
            shape=shape,
            dtype=dtype,
            value_summary=f"{sample}... dtype={dtype}"
        )
