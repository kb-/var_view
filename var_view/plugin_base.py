# var_view/plugin_base.py

from abc import ABC, abstractmethod


class PluginBase(ABC):
    """
    Abstract base class for all plugins.
    """

    @abstractmethod
    def register_handlers(self, register_type_handler):
        """
        Method to register type handlers.
        Each plugin must implement this method.
        """
        pass
