from PyQt6.QtWidgets import QApplication, QMainWindow, QTreeView, QVBoxLayout, QWidget, \
    QMenu, QHeaderView
from PyQt6.QtGui import QStandardItemModel, QStandardItem, QAction
from PyQt6.QtCore import QObject, Qt
import numpy as np
import torch
import sys
import inspect
import logging
from variableExporter import VariableExporter


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
        # self.tree_view.setDragDropMode(QTreeView.DragDropMode.DragOnly)
        self.tree_view.setSelectionMode(QTreeView.SelectionMode.ExtendedSelection)

        # Set resize mode to ResizeToContents for each column
        header = self.tree_view.header()
        for i in range(self.model.columnCount()):
            header.setSectionResizeMode(i, QHeaderView.ResizeMode.ResizeToContents)
            # Optionally, set minimum widths to prevent columns from being too narrow
            header.setMinimumSectionSize(100)  # Adjust as needed

        # Connect expand signal for lazy loading
        self.tree_view.expanded.connect(self.handle_expand)

        # Add context menu for exporting
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

            # Prevent editing and enable dragging
            for item in [item_name, item_type, item_value, item_memory]:
                item.setEditable(False)
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsDragEnabled)

            # Append items to the parent
            parent_item.appendRow([item_name, item_type, item_value, item_memory])

            # Add a placeholder for lazy loading if the variable can be expanded
            if lazy_load and self.can_expand(value):
                placeholder = QStandardItem("Loading...")
                # Enable dragging for the placeholder as well
                placeholder.setFlags(
                    placeholder.flags() | Qt.ItemFlag.ItemIsDragEnabled)
                item_name.appendRow([placeholder])

        except Exception as e:
            logging.error(f"Error adding variable {name}: {e}")
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
            placeholder = item.child(0)
            if placeholder.text() == "Loading...":
                # Remove the placeholder
                item.removeRow(0)
                # Resolve the variable and load its children
                variable_name = self.resolve_item_path(item)
                value = self.resolve_variable(variable_name, self.variables)
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
                            logging.error(f"Error accessing attribute {attr}: {e}")
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
            logging.error(f"Error loading children of {parent_item.text()}: {e}")
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

            parent_item.appendRow([method_item, method_type, method_value])
        except Exception as e:
            logging.error(f"Error adding method {name}: {e}")
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
                # Append list index directly without a dot
                path += part
            else:
                if path:
                    path += "." + part
                else:
                    path = part
        return path

    def resolve_variable(self, name, root_variables):
        """Resolve a variable name to its value."""
        try:
            components = name.split('.')
            value = root_variables
            for component in components:
                if component.startswith("[") and component.endswith("]"):
                    try:
                        index = int(component[1:-1])
                        value = value[index]
                    except (ValueError, IndexError, KeyError):
                        return None
                elif isinstance(value, dict) and component in value:
                    value = value[component]
                elif hasattr(value, component):
                    value = getattr(value, component)
                else:
                    return None
            return value
        except Exception as e:
            logging.error(f"Error resolving variable {name}: {e}")
            return None

    def format_value(self, value):
        """Format the value for display, truncating long values or sampling large
        data."""
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
        index = self.tree_view.indexAt(position)
        if index.isValid():
            menu = QMenu()
            export_action = QAction("Export", self)
            export_action.triggered.connect(lambda: self.export_variable(index))
            copy_path_action = QAction("Copy Path", self)
            copy_path_action.triggered.connect(lambda: self.copy_variable_path(index))
            menu.addAction(export_action)
            menu.addAction(copy_path_action)
            menu.exec(self.tree_view.viewport().mapToGlobal(position))

    def copy_variable_path(self, index):
        """Copy the full path of the selected variable to the clipboard."""
        # Always retrieve the item from column 0 (Variable column)
        variable_index = self.model.index(index.row(), 0, index.parent())
        item = self.model.itemFromIndex(variable_index)
        if item:
            path = self.resolve_item_path(item)
            clipboard = QApplication.clipboard()
            clipboard.setText(path)
            logging.debug(f"Copied to clipboard: {path}")

    def export_variable(self, index):
        item = self.model.itemFromIndex(index)
        variable_name = item.text()
        value = self.resolve_variable(variable_name, self.variables)
        if value is not None:
            self.exporter.export_variable(variable_name, value)


if __name__ == "__main__":
    # Configure logging for debugging purposes
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s - %(levelname)s - %(message)s')


    # Test data
    class TestClass:
        def __init__(self):
            self.attr1 = "Test Attribute"
            self.attr2 = 42

        def method1(self):
            return "Method 1"

        def method2(self):
            return "Method 2"


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Test data
    app_variables = {
        "list_var": [1, 2, 3, 4, 5],
        "nested_dict": {"a": 1, "b": {"c": 2, "d": 3}},
        "numpy_array": np.random.rand(100, 100),
        "torch_tensor": torch.rand(10, 10).to(device),
        "string_var": "Hello, World!",
        "test_obj": TestClass(),
    }
    app_variables.update({
        "huge_tensor": torch.rand(10000, 10000).to(device),
        "complex_nested_dict": {
            "level1": {"level2": {"level3": [1, 2, 3, 4, {"deep": "value"}]}}},
        "cyclic_ref": {}
    })
    app_variables["cyclic_ref"]["self"] = app_variables["cyclic_ref"]

    app = QApplication(sys.argv)
    viewer = VariableViewer(app_variables)
    viewer.show()
    sys.exit(app.exec())
