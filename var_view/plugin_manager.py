# var_view/plugin_manager.py

import importlib
import pkgutil
import os
import logging
from collections import OrderedDict

from var_view.plugin_base import PluginBase

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

class PluginManager:
    """
    Manages the discovery and loading of plugins.
    """

    def __init__(self):
        self.type_handlers = OrderedDict()  # Maintains insertion order for overriding

    def load_plugins_from_package(self, package):
        """
        Load built-in plugins from a given package.
        """
        logger.debug(f"Loading built-in plugins from package: {package.__name__}")
        for loader, module_name, is_pkg in pkgutil.iter_modules(package.__path__):
            full_module_name = f"{package.__name__}.{module_name}"
            try:
                module = importlib.import_module(full_module_name)
                self._register_plugin(module)
                logger.info(f"Loaded built-in plugin: {full_module_name}")
            except Exception as e:
                logger.error(f"Failed to load built-in plugin '{full_module_name}': {e}")

    def load_plugins_from_directory(self, directory, namespace='app_plugins'):
        """
        Load app-specific plugins from a specified directory.
        """
        logger.debug(f"Loading app-specific plugins from directory: {directory}")
        if not os.path.isdir(directory):
            logger.warning(f"Plugin directory '{directory}' does not exist.")
            return

        # Add the directory to sys.path to allow module imports
        if directory not in os.sys.path:
            os.sys.path.insert(0, directory)

        for filename in os.listdir(directory):
            if filename.endswith(".py") and not filename.startswith("_"):
                module_name = os.path.splitext(filename)[0]
                full_module_name = f"{namespace}.{module_name}"
                try:
                    module = importlib.import_module(module_name)
                    self._register_plugin(module)
                    logger.info(f"Loaded app-specific plugin: {full_module_name}")
                except Exception as e:
                    logger.error(f"Failed to load app-specific plugin '{full_module_name}': {e}")

    def _register_plugin(self, module):
        """
        Register a plugin by invoking its register_handlers method.
        """
        for attribute_name in dir(module):
            attribute = getattr(module, attribute_name)
            if isinstance(attribute, type) and issubclass(attribute, PluginBase) and attribute is not PluginBase:
                plugin_instance = attribute()
                plugin_instance.register_handlers(self.register_type_handler)
                logger.debug(f"Registered plugin handler from '{module.__name__}.{attribute_name}'")

    def register_type_handler(self, data_type, handler):
        """
        Register a custom handler for a specific data type.
        Handler should return a VariableRepresentation instance (or string fallback).
        """
        self.type_handlers[data_type] = handler
        logger.debug(f"Registered type handler for '{data_type}'")

    def get_handler_for_type(self, data):
        """
        Retrieve the handler for the given data based on its type.
        """
        for data_type, handler in self.type_handlers.items():
            if isinstance(data, data_type):
                return handler
        return None
