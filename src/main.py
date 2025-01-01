# main.py
# TODO: memory and lazy loading to improve. Wrong size on root. Display grayed with + sign before fully loaded?
# TODO: fix update. Doesn't work for SVA ROI
# TODO: tree doesn't display child object content in list (SVA ROIs)
# TODO: fix "Variable '{variable_name}' could not be resolved and was not exported." when trying to export a child of globals (tried on a frame returned by SVA reader)
import asyncio
import os
import shutil
import tempfile
import uuid
from uuid import uuid4

from PyQt6 import QtWidgets
from PyQt6.QtWidgets import QApplication, QMainWindow, QTreeView, QVBoxLayout, QWidget, \
    QMenu, QHeaderView, QMessageBox
from PyQt6.QtGui import QStandardItemModel, QStandardItem, QAction
from PyQt6.QtCore import QObject, Qt, QTimer
import numpy as np
import torch
import sys
import inspect
import logging
import re  # Added import for regex

from jupyter_client import AsyncKernelManager, KernelManager

from app.variableExporter import VariableExporter
from qtconsole.rich_jupyter_widget import RichJupyterWidget
from qtconsole.inprocess import QtInProcessKernelManager


class VariableStandardItemModel(QStandardItemModel):
    def __init__(self, viewer, parent=None):
        super().__init__(parent)
        self.viewer = viewer  # Reference to VariableViewer

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


class VariableViewer(QMainWindow):
    def __init__(self, variables):
        super().__init__()
        self.variables = variables
        self.exporter = VariableExporter(self)
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Variable Viewer")
        self.resize(1040, 800)

        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # Tree view
        self.tree_view = QTreeView()
        layout.addWidget(self.tree_view)

        # Set custom model
        self.model = VariableStandardItemModel(self)
        self.model.setHorizontalHeaderLabels(
            ["Variable", "Type", "Value", "Memory"]
        )
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

    def resize_all_columns(self):
        """Resize all columns to fit their contents."""
        for column in range(self.model.columnCount()):
            self.tree_view.resizeColumnToContents(column)

    def refresh_view(self):
        """
        Refresh the entire view by clearing and repopulating the model.
        This captures the current state of variables.
        """
        self.model.clear()
        self.model.setHorizontalHeaderLabels(
            ["Variable", "Type", "Value", "Memory"])
        for name, value in self.variables.items():
            self.add_variable(name, value, self.model.invisibleRootItem())
        # Adjust column sizes after initial load
        self.resize_all_columns()

    def add_variable(self, name, value, parent_item, lazy_load=True):
        try:
            # Determine the type
            value_type = type(value).__name__

            # Append dtype or equivalent if applicable
            if isinstance(value, np.ndarray):
                value_type += f" ({value.dtype})"
            elif isinstance(value, torch.Tensor):
                value_type += f" ({value.dtype})"

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
            bytes_size = 0
            if isinstance(value, np.ndarray):
                bytes_size = value.nbytes
            elif isinstance(value, torch.Tensor):
                bytes_size = value.element_size() * value.numel()
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
        return isinstance(value, (list, tuple, dict, QObject)) or hasattr(value,
                                                                          '__dict__')

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
                value = self.resolve_variable(path, self.variables)
                if value is not None:
                    self.load_children(item, value)
                    # After loading children, resize columns to fit new content
                    self.resize_all_columns()

    def load_children(self, parent_item, value):
        """Load and display children of a variable."""
        try:
            if isinstance(value, (list, tuple)):
                for i, sub_value in enumerate(value):
                    self.add_variable(f"[{i}]", sub_value, parent_item)
            elif isinstance(value, dict):
                for key, sub_value in value.items():
                    self.add_variable(str(key), sub_value, parent_item)
            elif isinstance(value, QObject):
                for attr in dir(value):
                    if not attr.startswith("_"):  # Skip private or special methods
                        try:
                            attr_value = getattr(value, attr)
                            if inspect.ismethod(attr_value):
                                self.add_method(attr, attr_value, parent_item)
                            else:
                                self.add_variable(attr, attr_value, parent_item)
                        except Exception as e:
                            logging.error(f"Error accessing attribute '{attr}': {e}")
                            self.add_variable(attr, f"<Error: {e}>", parent_item,
                                              lazy_load=False)
            elif hasattr(value, '__dict__'):
                for attr_name, attr_value in vars(value).items():
                    self.add_variable(attr_name, attr_value, parent_item)

                # Add methods separately for better organization
                for name, method in inspect.getmembers(value,
                                                       predicate=inspect.ismethod):
                    self.add_method(name, method, parent_item)

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

    def add_method(self, name, method, parent_item):
        """Add a method to the tree."""
        try:
            method_item = QStandardItem(name)
            method_type = QStandardItem("Method")
            method_value = QStandardItem(str(method))

            # Prevent editing and enable dragging
            for item in [method_item, method_type, method_value]:
                item.setEditable(False)
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsDragEnabled)

            # Append an empty item for the Memory column to maintain consistency
            parent_item.appendRow(
                [method_item, method_type, method_value, QStandardItem()])
        except Exception as e:
            logging.error(f"Error adding method '{name}': {e}")
            parent_item.appendRow([
                QStandardItem(name),
                QStandardItem("Error"),
                QStandardItem(f"<Error: {e}>"),
                QStandardItem("N/A")
            ])

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
            if part.startswith("[") and path:
                # Do NOT insert a dot before list indices
                path += part  # This results in "list_obj[0].car.owner.age"
            else:
                if path:
                    path += "." + part
                else:
                    path = part
        return path

    def resolve_variable(self, name, root_variables):
        """Resolve a variable name to its value."""
        try:
            # Use regex to split the path into components
            pattern = re.compile(r'\w+|\[\d+\]')
            components = pattern.findall(name)
            value = root_variables
            for comp in components:
                if comp.startswith("[") and comp.endswith("]"):
                    # It's a list or tuple index
                    index = int(comp[1:-1])
                    value = value[index]
                else:
                    # It's an attribute or dict key
                    if isinstance(value, dict):
                        value = value.get(comp, None)
                    else:
                        value = getattr(value, comp, None)
                if value is None:
                    return None
            return value
        except Exception as e:
            logging.error(f"Error resolving variable '{name}': {e}")
            return None

    def format_value(self, value):
        """Format the value for display, always sampling the first five elements of large data."""
        try:
            if isinstance(value, str):
                return value if len(value) <= 50 else value[:47] + "..."
            elif isinstance(value, (list, tuple, dict)):
                return f"{type(value).__name__}[{len(value)}]"
            elif isinstance(value, np.ndarray):
                flattened = value.flatten()
                # Sample first 5 elements
                sample = flattened[:5] if flattened.size >= 5 else flattened
                return f"ndarray{value.shape}: {sample.tolist()}..."
            elif isinstance(value, torch.Tensor):
                # Avoid accessing 'imag' for non-complex tensors
                if torch.is_complex(value):
                    sample = value.flatten().tolist()[:5]
                    return f"Tensor{tuple(value.shape)}: {sample}..."
                else:
                    flattened = value.flatten().tolist()
                    # Sample first 5 elements
                    sample = flattened[:5] if len(flattened) >= 5 else flattened
                    return f"Tensor{tuple(value.shape)}: {sample}..."
            elif hasattr(value, '__dict__'):  # Objects with attributes
                return "{...}"
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
        value = self.resolve_variable(variable_name, self.variables)
        if value is not None:
            self.exporter.export_variable(variable_name, value)
        else:
            QMessageBox.warning(self, "Export Warning",
                                f"Variable '{variable_name}' could not be resolved and was not exported.")

    def export_selected_variables(self, indexes):
        """Export multiple selected variables."""
        variables_to_export = {}
        for index in indexes:
            if index.column() != 0:
                continue  # Only process Variable column
            item = self.model.itemFromIndex(index)
            if item:
                path = self.resolve_item_path(item)
                value = self.resolve_variable(path, self.variables)
                if value is not None:
                    variables_to_export[path] = value

        if variables_to_export:
            self.exporter.export_variables(variables_to_export)
        else:
            QMessageBox.warning(self, "Export Warning",
                                "No valid variables selected for export.")

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
                current_value = self.resolve_variable(path, self.variables)
                if current_value is not None:
                    # Update Type, Value, and Memory columns only for root nodes
                    if item.parent() is None:
                        self.model.item(row, 1).setText(type(current_value).__name__)
                        formatted_value = self.format_value(current_value)
                        self.model.item(row, 2).setText(formatted_value)
                        memory_usage = self.calculate_memory_usage(current_value)
                        self.model.item(row, 3).setText(memory_usage)
                else:
                    # Handle cases where the variable could not be resolved
                    self.model.item(row, 2).setText("<Unavailable>")
                    self.model.item(row, 3).setText("N/A")

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

    def add_console(self, variables):
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
        console_window = QtWidgets.QWidget()
        console_window.setWindowTitle("Console")
        layout = QtWidgets.QVBoxLayout(console_window)
        layout.addWidget(console)
        console_window.resize(600, 960)  # Set initial size for the console window
        console_window.show()

        # Store a reference to prevent garbage collection
        self.console_window = console_window

        # Inject the shared app_variables dictionary into the console
        # kernel_manager.kernel.shell.push({"app_variables": variables})
        kernel = kernel_manager.kernel.shell
        kernel.push({"app_variables": variables})

        # Unpack app_variables into the kernel namespace
        for key, value in variables.items():
            kernel.push({key: value})

        logging.info("Console window opened and app_variables instance injected.")


if __name__ == "__main__":
    # Configure logging for debugging purposes
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')


    # Enhanced Test data with custom objects
    class Engine:
        def __init__(self, horsepower, type_):
            self.horsepower = horsepower
            self.type = type_

        def start(self):
            return "Engine started."


    class Car:
        def __init__(self, make, model, engine):
            self.make = make
            self.model = model
            self.engine = engine
            self.owner = None  # To be set later, creating a cyclic reference

        def drive(self):
            return f"Driving the {self.make} {self.model}."

        def set_owner(self, owner):
            self.owner = owner


    class Person:
        def __init__(self, name, age, car=None):
            self.name = name
            self.age = age
            self.car = car

        def greet(self):
            return f"Hello, my name is {self.name}."

        def buy_car(self, car):
            self.car = car
            car.set_owner(self)


    # Instantiate custom objects
    engine_v8 = Engine(450, "V8")
    car_ferrari = Car("Ferrari", "488 Spider", engine_v8)
    person_john = Person("John Doe", 30)

    # Establish cyclic reference
    person_john.buy_car(car_ferrari)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Test data
    app_variables = {
        "list_var": [1, 2, 3, 4, 5],
        "nested_dict": {"a": 1, "b": {"c": 2, "d": 3}},
        "numpy_array": np.random.rand(100, 100),
        "torch_tensor": torch.rand(10, 10).to(device),
        "string_var": "Try right click update!",
        "test_obj": person_john,  # Custom object
        "list_obj": [person_john, person_john, person_john],
    }
    app_variables.update({
        "huge_tensor": torch.rand(10000, 10000),
        "complex_nested_dict": {
            "level1": {"level2": {"level3": [1, 2, 3, 4, {"deep": "value"}]}}},
        "cyclic_ref": {}
    })
    app_variables["cyclic_ref"]["self"] = app_variables["cyclic_ref"]

    app = QApplication(sys.argv)
    app_variables = app_variables
    viewer = VariableViewer(app_variables)
    # Use the same `app_variables` reference in both methods
    viewer.add_console(app_variables)
    viewer.show()

    def update_variables_after_show(app_variables, person_john):
        """Whatever you want to do once the GUI is displayed."""
        app_variables["string_var"] = "Updated String"
        app_variables["nested_dict"]["a"] = "updated"
        # Modify custom object
        person_john.age = 31
        person_john.car.engine.horsepower = 500
        logging.info(f"update_variables_after_show")

    # Schedule the modifications to happen once the event loop starts
    QTimer.singleShot(5000, lambda: update_variables_after_show(app_variables, person_john))
    sys.exit(app.exec())
