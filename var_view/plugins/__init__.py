# var_view/plugins/__init__.py

import importlib
import pkgutil
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # Adjust as needed

plugins = {}


def import_plugins(register_type_handler):
    """
    Discover and import all plugins in the current package,
    then invoke their register_handlers functions.

    :param register_type_handler: Function to register type handlers.
    """
    package = importlib.import_module(__name__)
    package_path = package.__path__

    for loader, module_name, is_pkg in pkgutil.iter_modules(package_path):
        if is_pkg:
            logger.warning(
                f"Skipping sub-package '{module_name}' as plugins should be modules.")
            continue  # Skip sub-packages if any

        full_module_name = f"{package.__name__}.{module_name}"
        try:
            module = importlib.import_module(full_module_name)
            if hasattr(module, 'register_handlers'):
                module.register_handlers(register_type_handler)
                plugins[module_name] = module
                logger.info(f"Loaded and registered plugin: {full_module_name}")
            else:
                logger.warning(
                    f"Plugin '{full_module_name}' does not have a 'register_handlers' function.")
        except Exception as e:
            logger.error(f"Failed to load plugin '{full_module_name}': {e}")

    return plugins

# The plugins will be imported and registered from the main application
