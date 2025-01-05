# var_view/plugins/numpy_plugin.py
import numpy as np

from var_view.plugin_base import PluginBase
from var_view.variable_viewer import VariableRepresentation


class NumpyArrayPlugin(PluginBase):
    """
    Plugin to handle numpy.ndarray objects.
    """

    def register_handlers(self, register_type_handler):
        register_type_handler(np.ndarray, self.numpy_array_formatter)

    def numpy_array_formatter(self, value: np.ndarray) -> VariableRepresentation:
        shape = value.shape
        dtype = str(value.dtype)
        sample = value.flatten()[:5].tolist()  # First 5 elements
        return VariableRepresentation(
            nbytes=value.nbytes,
            shape=shape,
            dtype=dtype,
            value_summary=f"{sample}... dtype={dtype}"
        )
