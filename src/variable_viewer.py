# src/variable_viewer.py
import logging
import inspect
import re

import numpy as np
import torch
from PyQt6.QtWidgets import (
    QMainWindow, QTreeView, QVBoxLayout, QWidget, QMenu, QMessageBox,
    QHeaderView
)
from PyQt6.QtGui import QStandardItemModel, QStandardItem, QAction
from PyQt6.QtCore import Qt, QTimer, QObject
from variable_exporter import VariableExporter  # Ensure you have this module
from qtconsole.rich_jupyter_widget import RichJupyterWidget
from qtconsole.inprocess import QtInProcessKernelManager
from PyQt6.QtWidgets import QApplication


class VariableViewer(QMainWindow):
    def __init__(self, data_source):
        super().__init__()
        self.data_source = data_source  # Generic data source
        self.exporter = VariableExporter(self)
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Variable Viewer")
        self.resize(1060, 800)

        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # Tree view
        self.tree_view = QTreeView()
        layout.addWidget(self.tree_view)

        # Set custom model
        self.model = VariableStandardItemModel(self)
        self.model.setHorizontalHeaderLabels(["Variable", "Type", "Value", "Memory"])
        self.tree_view.setModel(self.model)

        # Enable dragging
        self.tree_view.setDragEnabled(True)
        self.tree_view.setDragDropMode(QTreeView.DragDropMode.DragOnly)
        self.tree_view.setSelectionMode(QTreeView.SelectionMode.ExtendedSelection)

        # Set resize mode to ResizeToContents for each column
        header = self.tree_view.header()
        for i in range(self.model.columnCount()):
            header.setSectionResizeMode(i, QHeaderView.ResizeMode.ResizeToContents)
            header.setMinimumSectionSize(50)  # Adjust as needed

        # Connect expand signal for lazy loading
        self.tree_view.expanded.connect(self.handle_expand)

        # Add context menu for exporting and updating
        self.tree_view.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.tree_view.customContextMenuRequested.connect(self.show_context_menu)

        # Load root variables
        self.refresh_view()

    def connect_signals(self):
        """
        Connect signals from the data source to the viewer.
        """
        if hasattr(self.data_source, 'variable_added'):
            self.data_source.variable_added.connect(self.on_variable_added)
        if hasattr(self.data_source, 'variable_updated'):
            self.data_source.variable_updated.connect(self.on_variable_updated)
        if hasattr(self.data_source, 'variable_removed'):
            self.data_source.variable_removed.connect(self.on_variable_removed)

    def resize_all_columns(self):
        """Resize all columns to fit their contents."""
        for column in range(self.model.columnCount()):
            self.tree_view.resizeColumnToContents(column)

    def refresh_view(self):
        """Refresh the view using the generic data source."""
        self.model.clear()
        self.model.setHorizontalHeaderLabels(["Variable", "Type", "Value", "Memory"])

        # Retrieve all top-level attributes/items from the data source
        all_vars = {
            name: getattr(self.data_source, name, None)
            # for name in dir(self.data_source) if not name.startswith('_')
            for name in dir(self.data_source)
            if not name.startswith('_') and not callable(
                getattr(self.data_source, name, None))
        }

        for name, value in all_vars.items():
            self.add_variable(name, value, self.model.invisibleRootItem())

    def add_variable(self, name, value, parent_item, lazy_load=True):
        try:
            # Determine the type
            value_type = type(value).__name__

            # Format the value nicely
            formatted_value = self.format_value(value)

            # Calculate memory usage
            memory_usage = self.calculate_memory_usage(value)

            # Create items for the columns
            item_name = QStandardItem(name)
            item_type = QStandardItem(value_type)
            item_value = QStandardItem(formatted_value)
            item_memory = QStandardItem(memory_usage)

            # Prevent editing and enable dragging only for Variable column
            item_name.setEditable(False)
            item_name.setFlags(item_name.flags() | Qt.ItemFlag.ItemIsDragEnabled)
            item_type.setEditable(False)
            item_value.setEditable(False)
            item_memory.setEditable(False)
            # Only Variable column is draggable as per flags method

            # Append items to the parent
            parent_item.appendRow([item_name, item_type, item_value, item_memory])

            # Conditional Debug Log
            if name in ["list_var", "nested_dict", "test_obj"]:  # Adjust as needed
                logging.debug(
                    f"Added variable '{name}' of type '{value_type}' with value '{formatted_value}'.")

            # Add a placeholder for lazy loading if the variable can be expanded
            if lazy_load and self.can_expand(value):
                placeholder = QStandardItem("Loading...")
                # Placeholder should not be draggable
                placeholder.setEditable(False)
                # Append a full row with placeholder and empty items for other columns
                item_name.appendRow(
                    [placeholder, QStandardItem(), QStandardItem(), QStandardItem()])
        except Exception as e:
            logging.error(f"Error adding variable '{name}': {e}")
            parent_item.appendRow([
                QStandardItem(name),
                QStandardItem("Error"),
                QStandardItem(f"<Error: {e}>"),
                QStandardItem("N/A")
            ])

    def calculate_memory_usage(self, value):
        """Calculate memory usage of a variable and format it with appropriate units."""
        try:
            if hasattr(value, 'nbytes'):  # Handles both torch.Tensor and np.ndarray
                bytes_size = value.nbytes
            else:
                # Fallback for standard Python objects
                import sys
                bytes_size = sys.getsizeof(value)

            # Format the size into appropriate units
            units = ['B', 'KB', 'MB', 'GB', 'TB']
            for unit in units:
                if bytes_size < 1024:
                    if unit == 'B':
                        return f"{bytes_size} {unit}"  # Avoid decimals for bytes
                    return f"{bytes_size:.2f} {unit}"
                bytes_size /= 1024
            return f"{bytes_size:.2f} PB"  # Fall back for very large sizes
        except Exception as e:
            logging.error(f"Error calculating memory usage: {e}")
            return "N/A"

    def can_expand(self, value):
        """Check if a variable can be expanded."""
        return isinstance(value, (dict, list, QObject)) or (
            hasattr(value, '__dict__') and not isinstance(value, (str, bytes))
        )

    def handle_expand(self, index):
        """Handle lazy loading when a tree item is expanded."""
        item = self.model.itemFromIndex(index)
        if item.hasChildren():
            # Get the first child
            placeholder = item.child(0, 0)
            if placeholder.text() == "Loading...":
                # Remove the placeholder
                item.removeRow(0)
                # Resolve the variable and load its children
                path = self.resolve_item_path(item)
                value = self.resolve_variable(path)
                if value is not None:
                    self.load_children(item, value)
                    # After loading children, resize columns to fit new content
                    self.resize_all_columns()

    def load_children(self, parent_item, value, visited=None):
        """Load and display children of a variable."""
        if visited is None:
            visited = set()

        try:
            if id(value) in visited:
                logging.debug(
                    f"Cyclic reference detected for '{parent_item.text()}'. Skipping further traversal.")
                self.add_variable("<Cyclic Reference>", "<Cyclic Reference>",
                                  parent_item, lazy_load=False)
                return
            visited.add(id(value))

            if isinstance(value, list):
                logging.debug(f"Loading children for list: {parent_item.text()}")
                for index, sub_value in enumerate(value):
                    self.add_variable(f"[{index}]", sub_value, parent_item)
            elif isinstance(value, dict):
                logging.debug(f"Loading children for dict: {parent_item.text()}")
                for key, sub_value in value.items():
                    self.add_variable(str(key), sub_value, parent_item)
            elif inspect.isclass(value):
                # Skip classes to prevent unwanted entries
                logging.debug(f"Skipping class '{value.__name__}'.")
                return
            elif hasattr(value, '__dict__') and not isinstance(value, QObject):
                # For non-QObject instances, iterate over __dict__
                logging.debug(f"Loading children for object: {parent_item.text()}")
                for attr_name, attr_value in vars(value).items():
                    if attr_name.startswith("_"):
                        logging.debug(f"Skipping private attribute '{attr_name}'.")
                        continue  # Skip private attributes
                    if callable(attr_value):
                        logging.debug(f"Skipping method '{attr_name}'.")
                        continue  # Skip methods
                    self.add_variable(attr_name, attr_value, parent_item)
            elif isinstance(value, QObject):
                # For QObject instances, iterate over properties without leading underscores and non-callable
                logging.debug(f"Loading children for QObject: {parent_item.text()}")
                for attr in dir(value):
                    if attr.startswith("_"):
                        logging.debug(f"Skipping private attribute '{attr}'.")
                        continue  # Skip private attributes
                    try:
                        attr_value = getattr(value, attr)
                        if callable(attr_value):
                            logging.debug(f"Skipping method '{attr}'.")
                            continue  # Skip methods
                        self.add_variable(attr, attr_value, parent_item)
                    except Exception as e:
                        logging.error(f"Error accessing attribute '{attr}': {e}")
                        self.add_variable(attr, f"<Error: {e}>", parent_item,
                                          lazy_load=False)
            # Add more type handlers if necessary

            # After loading children, adjust column sizes
            self.resize_all_columns()

        except Exception as e:
            logging.error(f"Error loading children of '{parent_item.text()}': {e}")
            parent_item.appendRow([
                QStandardItem("Error"),
                QStandardItem("Error"),
                QStandardItem(f"<Error: {e}>"),
                QStandardItem("N/A")
            ])

    def format_value(self, value):
        """Format the value for display, sampling the first five elements of large data."""
        try:
            if isinstance(value, str):
                return value if len(value) <= 50 else value[:47] + "..."
            elif isinstance(value, (list, dict)):
                return f"{type(value).__name__}({len(value)})"
            elif isinstance(value, torch.Tensor):
                # Handle Torch Tensors
                shape = tuple(value.shape)
                dtype = value.dtype
                sample = value.flatten().tolist()[:5]  # Sample first 5 elements
                return f"Tensor{shape} ({dtype}): {sample}..."
            elif isinstance(value, np.ndarray):
                # Handle NumPy Arrays
                shape = value.shape
                dtype = value.dtype
                sample = value.flatten()[:5].tolist()  # Sample first 5 elements
                return f"ndarray{shape} ({dtype}): {sample}..."
            elif hasattr(value, '__dict__') and not isinstance(value, QObject):
                return f"<{type(value).__name__}>"
            else:
                return str(value)
        except Exception as e:
            logging.error(f"Error formatting value: {e}")
            return "<Error>"

    def show_context_menu(self, position):
        indexes = self.tree_view.selectedIndexes()
        if not indexes:
            return  # No selection, do nothing

        menu = QMenu()

        # Add Export action for single or multiple selections
        if len(indexes) == 1:  # Single selection
            export_action = QAction("Export", self)
            export_action.triggered.connect(lambda: self.export_variable(indexes[0]))
            menu.addAction(export_action)
        elif len(indexes) > 1:  # Multiple selections
            export_selected_action = QAction("Export Selected", self)
            export_selected_action.triggered.connect(
                lambda: self.export_selected_variables(indexes))
            menu.addAction(export_selected_action)

        # Add Update action (root-level variables only)
        for index in indexes:
            item = self.model.itemFromIndex(index)
            if item.parent() is None:  # Root-level variable
                update_action = QAction("Update", self)
                update_action.triggered.connect(
                    lambda: self.update_selected_variables(indexes))
                menu.addAction(update_action)
                break

        # Add Copy Path action for any selection
        copy_path_action = QAction("Copy Path", self)
        copy_path_action.triggered.connect(lambda: self.copy_variable_path(indexes))
        menu.addAction(copy_path_action)

        menu.exec(self.tree_view.viewport().mapToGlobal(position))

    def export_variable(self, index):
        """Export a single selected variable."""
        item = self.model.itemFromIndex(index)
        variable_name = item.text()
        value = self.resolve_variable(variable_name)
        if value is not None:
            self.exporter.export_variable(variable_name, value)
        else:
            QMessageBox.warning(
                self, "Export Warning",
                f"Variable '{variable_name}' could not be resolved and was not exported."
            )

    def export_selected_variables(self, indexes):
        """Export multiple selected variables."""
        variables_to_export = {}
        for index in indexes:
            if index.column() != 0:
                continue  # Only process Variable column
            item = self.model.itemFromIndex(index)
            if item:
                path = self.resolve_item_path(item)
                value = self.resolve_variable(path)
                if value is not None:
                    variables_to_export[path] = value

        if variables_to_export:
            self.exporter.export_variables(variables_to_export)
        else:
            QMessageBox.warning(
                self, "Export Warning",
                "No valid variables selected for export."
            )

    def update_selected_variables(self, indexes):
        """
        Update root-level variables by reloading their entries in the tree.
        """
        for index in indexes:
            if index.column() == 0:  # Only process the "Variable" column
                item = self.model.itemFromIndex(index)
                if item and item.parent() is None:  # Check for root-level item
                    path = item.text()
                    value = getattr(self.data_source, path,
                                    None)  # Use `data_source` instead
                    if value is not None:
                        self.unload_and_reload_item(item, value, path)
                    else:
                        logging.warning(f"Root variable '{path}' is unavailable.")

    def unload_and_reload_item(self, item, value, path):
        """
        Unload and reload a tree item to reflect updated values.
        """
        try:
            logging.debug(f"Unloading and reloading item '{path}' with value: {value}")

            # Update parent item's Type, Value, and Memory columns
            item_type = type(value).__name__
            formatted_value = self.format_value(value)
            memory_usage = self.calculate_memory_usage(value)

            self.model.itemFromIndex(item.index().sibling(item.row(), 1)).setText(
                item_type)  # Type column
            self.model.itemFromIndex(item.index().sibling(item.row(), 2)).setText(
                formatted_value)  # Value column
            self.model.itemFromIndex(item.index().sibling(item.row(), 3)).setText(
                memory_usage)  # Memory column

            logging.info(f"Updated columns for item '{path}'.")

            # Clear existing children
            item.removeRows(0, item.rowCount())
            logging.info(f"Unloaded children for variable '{path}'.")

            # Reload children if the variable can be expanded
            if self.can_expand(value):
                self.load_children(item, value)
                logging.info(f"Reloaded children for variable '{path}'.")
            else:
                logging.info(f"Variable '{path}' is not expandable.")
        except Exception as e:
            logging.error(f"Error unloading and reloading item '{path}': {e}")

    def update_all_references(self, item, value, path):
        """
        Update all tree items referencing the same object.
        """
        try:
            # Iterate through all rows in the model
            for row in range(self.model.rowCount()):
                sibling_item = self.model.item(row, 0)  # Get the variable tree item
                sibling_path = self.resolve_item_path(sibling_item)
                sibling_value = self.resolve_variable(sibling_path)

                # Check if the sibling references the same object
                if sibling_value is value:
                    self.update_tree_item(sibling_item, sibling_value, sibling_path)
                    logging.info(f"Updated display for shared object '{path}'.")

            # Update the originally selected item
            self.update_tree_item(item, value, path)
        except Exception as e:
            logging.error(f"Error updating all references for '{path}': {e}")

    def update_tree_item(self, item, value, path):
        """
        Update the tree item and its sub-items based on the new value.
        """
        try:
            # Update Type, Value, and Memory columns
            self.model.itemFromIndex(
                item.index().sibling(item.index().row(), 1)).setText(
                type(value).__name__)
            formatted_value = self.format_value(value)
            self.model.itemFromIndex(
                item.index().sibling(item.index().row(), 2)).setText(formatted_value)
            memory_usage = self.calculate_memory_usage(value)
            self.model.itemFromIndex(
                item.index().sibling(item.index().row(), 3)).setText(memory_usage)
            logging.info(f"Updated display for variable '{path}'.")

            # Clear existing children
            item.removeRows(0, item.rowCount())

            # Reload children if the variable is expandable
            if self.can_expand(value):
                self.load_children(item, value)
        except Exception as e:
            logging.error(f"Error updating tree item '{path}': {e}")

    def copy_variable_path(self, indexes):
        """Copy the full path of the selected variables to the clipboard."""
        paths = []
        for index in indexes:
            if index.column() != 0:
                continue  # Only process Variable column
            item = self.model.itemFromIndex(index)
            if item:
                path = self.resolve_item_path(item)
                paths.append(path)

        if paths:
            clipboard = QApplication.clipboard()
            clipboard.setText("\n".join(paths))
            logging.debug(f"Copied to clipboard: {paths}")

    def add_console(self):
        # Create an in-process kernel manager
        kernel_manager = QtInProcessKernelManager()
        kernel_manager.start_kernel()
        kernel_manager.kernel.gui = "qt"

        # Create a kernel client and start channels
        kernel_client = kernel_manager.client()
        kernel_client.start_channels()

        # Create a Rich Jupyter Widget
        console = RichJupyterWidget()
        console.kernel_manager = kernel_manager
        console.kernel_client = kernel_client

        # Create a separate window for the console
        console_window = QWidget()
        console_window.setWindowTitle("Console")
        layout = QVBoxLayout(console_window)
        layout.addWidget(console)
        console_window.resize(600, 960)  # Set initial size for the console window
        console_window.show()

        # Store a reference to prevent garbage collection
        self.console_window = console_window

        # Inject the data source into the console's namespace
        kernel = kernel_manager.kernel.shell
        kernel.push({"data_source": self.data_source})  # Use `data_source` here

        logging.info("Console window opened and data source injected.")

    # Signal Handlers
    def on_variable_added(self, name):
        """Handle a new variable being added."""
        logging.info(f"Signal received: variable_added('{name}')")
        value = getattr(self.data_source, name, None)
        self.add_variable(name, value, self.model.invisibleRootItem())

    def on_variable_updated(self, name):
        """Handle a variable being updated."""
        logging.info(f"Signal received: variable_updated('{name}')")
        # Find the top-level item matching the variable_name
        root = self.model.invisibleRootItem()
        for row in range(root.rowCount()):
            item = root.child(row, 0)
            if item.text() == name:
                # Update Type, Value, and Memory columns
                try:
                    value = getattr(self.data_source, name, None)
                    if value is not None:
                        logging.debug(
                            f"Updating variable '{name}' with new value: {value}")
                        item.setText(type(value).__name__)  # Update type
                        formatted_value = self.format_value(value)
                        self.model.item(row, 2).setText(formatted_value)
                        memory_usage = self.calculate_memory_usage(value)
                        self.model.item(row, 3).setText(memory_usage)
                        logging.info(f"Updated display for variable '{name}'.")
                    else:
                        self.model.item(row, 2).setText("<Unavailable>")
                        self.model.item(row, 3).setText("N/A")
                        logging.warning(f"Variable '{name}' is now unavailable.")
                except AttributeError as e:
                    logging.error(f"Error accessing '{name}' in data source: {e}")
                break
        else:
            logging.warning(f"Variable '{name}' not found in the viewer.")

    def on_variable_removed(self, name):
        """Handle a variable being removed."""
        logging.info(f"Signal received: variable_removed('{name}')")
        # Find the top-level item matching the variable_name and remove it
        root = self.model.invisibleRootItem()
        for row in range(root.rowCount()):
            item = root.child(row, 0)
            if item.text() == name:
                self.model.removeRow(row)
                logging.info(f"Removed variable '{name}' from the viewer.")
                break
        else:
            logging.warning(f"Variable '{name}' not found in the viewer.")

    def resolve_variable(self, path):
        """Resolve a variable path to its value in a nested structure."""
        try:
            # Pattern to match attributes (.attr), dictionary keys (["key"]), and list indices ([index])
            pattern = re.compile(r'\w+|\.\w+|\["[^"]*"\]|\[\d+\]')
            components = pattern.findall(path)

            if not components:
                logging.error(f"Invalid path: {path}")
                return None

            # Extract the root component (the starting variable or attribute)
            root_component = components[0]
            value = getattr(self.data_source, root_component, None)
            if value is None:
                logging.error(f"Root variable '{root_component}' not found.")
                return None

            # Iterate through the remaining path components to resolve nested values
            for comp in components[1:]:
                if comp.startswith("[") and comp.endswith(
                        "]"):  # Handles list indices and dictionary keys
                    if '"' in comp:  # Dictionary key
                        key = comp.strip('["]')
                        if isinstance(value, dict):
                            value = value.get(key, None)
                            if value is None:
                                logging.error(
                                    f"Key '{key}' not found in dictionary at '{root_component}'.")
                                return None
                        else:
                            logging.error(
                                f"Expected a dictionary, got {type(value).__name__} for key access '{key}'.")
                            return None
                    else:  # List index
                        try:
                            index = int(comp.strip("[]"))
                            if isinstance(value, list):
                                if 0 <= index < len(value):
                                    value = value[index]
                                else:
                                    logging.error(
                                        f"Index {index} out of range for list at '{root_component}'.")
                                    return None
                            else:
                                logging.error(
                                    f"Expected a list, got {type(value).__name__} for index access '{index}'.")
                                return None
                        except ValueError:
                            logging.error(
                                f"Invalid list index '{comp}' in path '{path}'.")
                            return None
                elif comp.startswith("."):  # Object attribute
                    attr = comp.strip(".")
                    if hasattr(value, attr):
                        value = getattr(value, attr, None)
                        if value is None:
                            logging.error(
                                f"Attribute '{attr}' of '{path}' resolved to None.")
                            return None
                    else:
                        logging.error(
                            f"Attribute '{attr}' not found in object of type {type(value).__name__}.")
                        return None
                else:  # Unrecognized path component
                    logging.error(
                        f"Unexpected path component '{comp}' in path '{path}'.")
                    return None

            return value
        except Exception as e:
            logging.error(f"Error resolving variable '{path}': {e}")
            return None

    def resolve_item_path(self, item):
        """Get the full path of the variable represented by the tree item."""
        parts = []
        current_item = item
        while current_item is not None:
            parts.append(current_item.text())
            current_item = current_item.parent()
        # Reverse to get path from root to the item
        parts.reverse()

        path = ""
        value = None
        for i, part in enumerate(parts):
            if i == 0:
                # Root variable
                path = part
                value = getattr(self.data_source, part, None)
            else:
                if isinstance(value, list):
                    # part should be [index]
                    path += part  # part is like [0]
                    try:
                        index = int(part.strip("[]"))
                        value = value[index]
                    except (ValueError, IndexError, TypeError):
                        value = None
                elif isinstance(value, dict):
                    # part is a key
                    # Escape quotes in key if necessary
                    escaped_key = part.replace('"', '\\"').replace("'", "\\'")
                    path += f'["{escaped_key}"]'
                    value = value.get(part, None)
                elif hasattr(value, part):
                    # part is an attribute
                    path += f'.{part}'
                    value = getattr(value, part, None)
                else:
                    # Unknown type, default to dot notation
                    path += f'.{part}'
                    value = getattr(value, part, None)
        return path


class VariableStandardItemModel(QStandardItemModel):
    def __init__(self, viewer, parent=None, root_variable='variables'):
        super().__init__(parent)
        self.viewer = viewer  # Reference to VariableViewer
        self.root_variable = root_variable

    def supportedDragActions(self):
        return Qt.DropAction.CopyAction

    def mimeData(self, indexes):
        mime_data = super().mimeData(indexes)
        paths = []
        for index in indexes:
            if index.column() != 0:
                # Skip non-variable columns
                continue
            item = self.itemFromIndex(index)
            if item:
                path = self.viewer.resolve_item_path(item)
                if not path.startswith(self.root_variable):
                    path = f"{self.root_variable}.{path}"
                paths.append(path)
        if paths:
            mime_data.setText("\n".join(paths))
            logging.debug(f"Dragging variable paths: {paths}")
        return mime_data

    def flags(self, index):
        if not index.isValid():
            return Qt.ItemFlag.NoItemFlags
        if index.column() == 0:
            return super().flags(index) | Qt.ItemFlag.ItemIsDragEnabled
        else:
            return super().flags(index)


