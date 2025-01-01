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
    def __init__(self, variables_instance):
        super().__init__()
        self.variables_instance = variables_instance  # Instance of Variables
        self.exporter = VariableExporter(self)
        self.initUI()
        self.connect_signals()

    def initUI(self):
        self.setWindowTitle("Variable Viewer")
        self.resize(800, 600)

        # Menu Bar
        menubar = self.menuBar()
        view_menu = menubar.addMenu('View')

        # "Update" Action
        update_action = QAction('Update', self)
        update_action.setShortcut('Ctrl+R')
        update_action.setStatusTip('Refresh the Variable Viewer')
        update_action.triggered.connect(self.refresh_view)  # Connect to refresh_view
        view_menu.addAction(update_action)

        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # Tree view
        self.tree_view = QTreeView()
        layout.addWidget(self.tree_view)

        # Set custom model
        self.model = QStandardItemModel()
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
            header.setMinimumSectionSize(100)  # Adjust as needed

        # Connect expand signal for lazy loading
        self.tree_view.expanded.connect(self.handle_expand)

        # Add context menu for exporting and updating
        self.tree_view.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.tree_view.customContextMenuRequested.connect(self.show_context_menu)

        # Load root variables
        self.refresh_view()

        # After loading data, adjust column sizes
        self.resize_all_columns()

    def connect_signals(self):
        """
        Connect signals from Variables instance to the viewer.
        """
        self.variables_instance.variable_added.connect(self.on_variable_added)
        self.variables_instance.variable_updated.connect(self.on_variable_updated)
        self.variables_instance.variable_removed.connect(self.on_variable_removed)

    def resize_all_columns(self):
        """Resize all columns to fit their contents."""
        for column in range(self.model.columnCount()):
            self.tree_view.resizeColumnToContents(column)

    def refresh_view(self):
        """
        Refresh the entire view by clearing and repopulating the model.
        This captures the current state of variables.
        """
        logging.info("refresh_view() called: Refreshing Variable Viewer...")

        # Optional: Remove previous tree items
        self.model.clear()
        self.model.setHorizontalHeaderLabels(["Variable", "Type", "Value", "Memory"])

        all_vars = self.variables_instance.get_all_variables()
        logging.debug(f"All variables to load: {list(all_vars.keys())}")

        for name, value in all_vars.items():
            self.add_variable(name, value, self.model.invisibleRootItem())

        # Adjust column sizes after initial load
        self.resize_all_columns()
        logging.info("refresh_view() completed: Variable Viewer refreshed.")

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
        return isinstance(value, (dict, list, QObject))

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
        """Format the value for display, always sampling the first five elements of large data."""
        try:
            if isinstance(value, str):
                return value if len(value) <= 50 else value[:47] + "..."
            elif isinstance(value, (list, dict)):
                return f"{type(value).__name__}({len(value)})"
            elif isinstance(value, QObject):
                return f"<{type(value).__name__}>"
            elif isinstance(value, (np.ndarray, torch.Tensor)):
                if isinstance(value, np.ndarray):
                    shape = value.shape
                    sample = value.flatten()[:5].tolist()
                else:
                    shape = tuple(value.shape)
                    sample = value.flatten()[:5].tolist()
                return f"{type(value).__name__}{shape}: {sample}..."
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

        # Determine selected variables (only unique rows in the Variable column)
        selected_rows = set()
        for index in indexes:
            if index.column() == 0:
                selected_rows.add(index.row())

        if len(selected_rows) == 1:
            # Single selection: Enable Export
            export_action = QAction("Export", self)
            export_action.triggered.connect(lambda: self.export_variable(indexes[0]))
            menu.addAction(export_action)
        elif len(selected_rows) > 1:
            # Multiple selections: Enable Export Selected
            export_selected_action = QAction("Export Selected", self)
            export_selected_action.triggered.connect(
                lambda: self.export_selected_variables(indexes))
            menu.addAction(export_selected_action)
        else:
            # No valid selections for export
            export_action = QAction("Export", self)
            export_action.setEnabled(False)
            menu.addAction(export_action)

        # Add Update action (supports multiple selections)
        update_action = QAction("Update", self)
        update_action.triggered.connect(lambda: self.update_selected_variables(indexes))
        menu.addAction(update_action)

        # Add Copy Path action (supports multiple selections)
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
        Update the selected variables by re-fetching their current values.
        Supports multiple selections.
        """
        # Extract unique rows (variables) from selected indexes
        selected_rows = set()
        for index in indexes:
            if index.column() == 0:  # Variable column
                selected_rows.add(index.row())

        # Iterate over unique selected rows
        for row in selected_rows:
            item = self.model.item(row, 0)  # Get the Variable column item
            if item:
                path = self.resolve_item_path(item)
                value = self.resolve_variable(path)
                if value is not None:
                    # Update Type, Value, and Memory columns only for root nodes
                    if item.parent() is None:
                        self.model.item(row, 1).setText(type(value).__name__)
                        formatted_value = self.format_value(value)
                        self.model.item(row, 2).setText(formatted_value)
                        memory_usage = self.calculate_memory_usage(value)
                        self.model.item(row, 3).setText(memory_usage)
                        logging.info(f"Updated variable '{path}'.")
                else:
                    # Handle cases where the variable could not be resolved
                    self.model.item(row, 2).setText("<Unavailable>")
                    self.model.item(row, 3).setText("N/A")
                    logging.warning(f"Variable '{path}' is unavailable.")

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

    def add_console(self, variables_instance):
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

        # Inject the Variables instance into the console's namespace
        kernel = kernel_manager.kernel.shell
        kernel.push({"variables": variables_instance})

        logging.info("Console window opened and Variables instance injected.")

    # Signal Handlers
    def on_variable_added(self, name):
        """Handle a new variable being added."""
        logging.info(f"Signal received: variable_added('{name}')")
        value = self.variables_instance.get_variable(name)
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
                value = self.variables_instance.get_variable(name)
                if value is not None:
                    self.model.item(row, 1).setText(type(value).__name__)
                    formatted_value = self.format_value(value)
                    self.model.item(row, 2).setText(formatted_value)
                    memory_usage = self.calculate_memory_usage(value)
                    self.model.item(row, 3).setText(memory_usage)
                    logging.info(f"Updated display for variable '{name}'.")
                else:
                    self.model.item(row, 2).setText("<Unavailable>")
                    self.model.item(row, 3).setText("N/A")
                    logging.warning(f"Variable '{name}' is now unavailable.")
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

    def resolve_item_path(self, item):
        """Get the full path of the variable represented by the tree item."""
        parts = []
        current_item = item
        while current_item is not None:
            parts.append(current_item.text())
            current_item = current_item.parent()
        # Reverse to get path from root to the item
        parts.reverse()

        # Build the path, handling list indices appropriately
        path = ""
        for part in parts:
            if re.match(r'\[\d+\]', part):
                # It's a list index, append without dot
                path += part
            else:
                if path:
                    path += "." + part
                else:
                    path = part
        return path

    def resolve_variable(self, path):
        """Resolve a variable path to its value."""
        try:
            # Split the path into components
            pattern = re.compile(r'\w+|\[\d+\]')
            components = pattern.findall(path)
            if not components:
                return None
            value = self.variables_instance.get_variable(components[0])
            for comp in components[1:]:
                if comp.startswith("[") and comp.endswith("]"):
                    # It's a list index
                    index = int(comp[1:-1])
                    if isinstance(value, list):
                        if 0 <= index < len(value):
                            value = value[index]
                        else:
                            return None
                    else:
                        return None
                else:
                    # It's an attribute or dict key
                    if isinstance(value, dict):
                        value = value.get(comp, None)
                    elif isinstance(value, list):
                        if comp.isdigit():
                            idx = int(comp)
                            value = value[idx] if idx < len(value) else None
                        else:
                            return None
                    elif hasattr(value, comp):
                        value = getattr(value, comp, None)
                    else:
                        return None
                if value is None:
                    return None
            return value
        except Exception as e:
            logging.error(f"Error resolving variable '{path}': {e}")
            return None
