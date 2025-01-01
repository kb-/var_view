# src/variables.py
from PyQt6.QtCore import QObject, pyqtSignal
import logging


class Variables(QObject):
    """
    A centralized class to manage application variables.
    Emits signals when variables are added, updated, or removed.
    """
    variable_added = pyqtSignal(str)
    variable_updated = pyqtSignal(str)
    variable_removed = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self._variables = {}

    def add_variable(self, name, value):
        """
        Add a new variable to the manager.
        """
        if name in self._variables:
            logging.warning(f"Variable '{name}' already exists. Use update_variable to modify it.")
            return
        self._variables[name] = value
        self.variable_added.emit(name)
        logging.info(f"Added variable '{name}'.")

    def update_variable(self, name, value):
        """
        Update an existing variable.
        """
        if name not in self._variables:
            logging.warning(f"Variable '{name}' does not exist. Use add_variable to add it.")
            return
        self._variables[name] = value
        self.variable_updated.emit(name)
        logging.info(f"Updated variable '{name}'.")

    def remove_variable(self, name):
        """
        Remove a variable from the manager.
        """
        if name in self._variables:
            del self._variables[name]
            self.variable_removed.emit(name)
            logging.info(f"Removed variable '{name}'.")
        else:
            logging.warning(f"Variable '{name}' does not exist.")

    def get_variable(self, name):
        """
        Retrieve a variable's value.
        """
        return self._variables.get(name, None)

    def get_all_variables(self):
        """
        Retrieve all variables.
        """
        return self._variables.copy()

    def set_variable(self, name, value):
        """
        Set a variable (add or update).
        """
        if name in self._variables:
            self.update_variable(name, value)
        else:
            self.add_variable(name, value)

    def __getattr__(self, name):
        """
        Allow attribute-like access to variables.
        """
        if name in self._variables:
            return self._variables[name]
        raise AttributeError(f"'Variables' object has no attribute '{name}'")

    def __setattr__(self, name, value):
        """
        Allow setting variables as attributes.
        Emit appropriate signals on changes.
        """
        # Prevent recursion for internal attributes
        if name in ["_variables", "variable_added", "variable_updated", "variable_removed"]:
            super().__setattr__(name, value)
        else:
            if name in self._variables:
                self.update_variable(name, value)
                logging.info(f"Updated variable '{name}' via attribute.")
            else:
                self.add_variable(name, value)
                logging.info(f"Added variable '{name}' via attribute.")

    def __dir__(self):
        """
        Customize the list of attributes returned by dir().
        Includes only the dynamic variables.
        """
        return list(self._variables.keys())
