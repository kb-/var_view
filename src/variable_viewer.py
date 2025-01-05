# src/variable_viewer.py

import importlib
import logging
import inspect
import os
import re
import sys  # Added import for sys.getsizeof

from PyQt6.QtWidgets import (
    QMainWindow, QTreeView, QVBoxLayout, QWidget, QMenu, QMessageBox,
    QHeaderView, QApplication
)
from PyQt6.QtGui import QStandardItemModel, QStandardItem, QAction
from PyQt6.QtCore import Qt, QObject

# Console imports
from qtconsole.rich_jupyter_widget import RichJupyterWidget
from qtconsole.inprocess import QtInProcessKernelManager

from variable_exporter import VariableExporter

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

# Global registry of plugin-based type handlers
type_handlers = {}


def register_type_handler(data_type, handler):
    """
    Register a custom handler for a specific data type.
    Handler should return a VariableRepresentation instance (or string fallback).
    """
    type_handlers[data_type] = handler


class VariableRepresentation:
    """
    Encapsulates metadata about a variable that a plugin might provide:
      - nbytes: memory usage
      - shape: shape of array-like data
      - dtype: data type
      - value_summary: a short text summarizing the contents or first elements
    """
    def __init__(self, nbytes, shape=None, dtype=None, value_summary=None):
        self.nbytes = nbytes
        self.shape = shape
        self.dtype = dtype
        self.value_summary = value_summary  # e.g. "sample=[1,2,3]..."

    def __str__(self):
        """
        Default text representation combining shape, dtype, and value summary.
        """
        parts = []
        if self.shape:
            parts.append(f"shape={self.shape}")
        if self.dtype:
            parts.append(f"dtype={self.dtype}")
        if self.value_summary:
            parts.append(str(self.value_summary))
        return ", ".join(parts) if parts else "N/A"


def infer_type_hint_general(data) -> str:
    """
    Dynamically infers a type hint for the given input data.
    Handles first-level types for nested structures, dicts, lists, sets, etc.
    For example: dict[str, int|dict], list[int|float], etc.
    """
    from collections.abc import Iterable

    if isinstance(data, dict):
        # Empty dict
        if len(data) == 0:
            return "dict"
        key_types = {type(k).__name__ for k in data.keys()}
        value_types = {type(v).__name__ for v in data.values()}
        combined_keys = " | ".join(sorted(key_types))
        combined_values = " | ".join(sorted(value_types))
        return f"dict[{combined_keys}, {combined_values}]"

    elif isinstance(data, Iterable) and not isinstance(data, (str, bytes)):
        try:
            if len(data) == 0:
                return type(data).__name__  # e.g. "list"
        except TypeError:
            # Some iterables don't support len()
            pass
        # For a list/tuple/set, gather the unique first-level element types
        element_types = {type(element).__name__ for element in data}
        combined_type = " | ".join(sorted(element_types))
        return f"{type(data).__name__}[{combined_type}]"

    else:
        # For a non-iterable or str/bytes, just return the type name
        return type(data).__name__


class VariableViewer(QMainWindow):
    """
    A GUI-based viewer that:
      - Displays variables in a tree (with columns: Variable, Type, Size, Value, Memory)
      - Supports plugins for data-type-specific handling
      - Can expand nested objects or lists/dicts (lazy loading)
      - Includes a console integration for interactive usage
    """
    def __init__(self, data_source, alias="data_source", plugin_dir=None):
        super().__init__()
        self.data_source = data_source
        self.exporter = VariableExporter(self)

        # Load plugins from default and/or custom directories
        self.load_plugins(plugin_dir)

        self.initUI(alias)

    def initUI(self, alias):
        self.setWindowTitle("Variable Viewer")
        self.resize(1200, 800)

        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # Tree view
        self.tree_view = QTreeView()
        layout.addWidget(self.tree_view)

        # Set custom model
        self.model = VariableStandardItemModel(self, alias)
        self.model.setHorizontalHeaderLabels(["Variable", "Type", "Size", "Value", "Memory"])
        self.tree_view.setModel(self.model)

        # Tree properties
        self.tree_view.setDragEnabled(True)
        self.tree_view.setDragDropMode(QTreeView.DragDropMode.DragOnly)
        self.tree_view.setSelectionMode(QTreeView.SelectionMode.ExtendedSelection)
        self.tree_view.expanded.connect(self.handle_expand)
        self.tree_view.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.tree_view.customContextMenuRequested.connect(self.show_context_menu)

        # Resize columns
        header = self.tree_view.header()
        for i in range(self.model.columnCount()):
            header.setSectionResizeMode(i, QHeaderView.ResizeMode.ResizeToContents)
            header.setMinimumSectionSize(50)

        # Initial load
        self.refresh_view()

        # Ensure columns resize after initial load
        self.resize_all_columns()

    @staticmethod
    def load_plugins(plugin_dir=None):
        """
        Load plugins from the default "plugins" dir and an optional custom plugin_dir.
        Each plugin calls register_handlers(...) for specialized data types.
        """
        plugin_dirs = set()

        # Default plugin directory (relative to this file)
        default_plugin_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "plugins"))
        plugin_dirs.add(default_plugin_dir)

        # Add custom plugin directory if provided (and different from default)
        if plugin_dir:
            abs_plugin_dir = os.path.abspath(plugin_dir)
            if abs_plugin_dir != default_plugin_dir:
                plugin_dirs.add(abs_plugin_dir)

        for directory in plugin_dirs:
            if not os.path.exists(directory):
                logger.warning(f"Plugin directory '{directory}' does not exist.")
                continue
            for filename in os.listdir(directory):
                if filename.endswith(".py") and not filename.startswith("_"):
                    plugin_path = os.path.join(directory, filename)
                    try:
                        spec = importlib.util.spec_from_file_location("plugin", plugin_path)
                        plugin = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(plugin)

                        if hasattr(plugin, "register_handlers"):
                            plugin.register_handlers(register_type_handler)
                            logger.info(f"Loaded plugin: {filename}")
                        else:
                            logger.info(f"Skipped {filename}: No register_handlers function")
                    except Exception as e:
                        logger.error(f"Failed to load plugin {filename}: {e}")

    def resize_all_columns(self):
        """Resize all columns to fit their contents."""
        for column in range(self.model.columnCount()):
            self.tree_view.resizeColumnToContents(column)

    def refresh_view(self):
        """
        Clears the current model and repopulates the top-level variables
        from self.data_source.
        """
        self.model.clear()
        self.model.setHorizontalHeaderLabels(["Variable", "Type", "Size", "Value", "Memory"])

        all_vars = {
            name: getattr(self.data_source, name, None)
            for name in dir(self.data_source)
            if not name.startswith("_") and not callable(getattr(self.data_source, name, None))
        }

        for name, value in all_vars.items():
            self.add_variable(name, value, self.model.invisibleRootItem())

    def add_variable(self, name, value, parent_item, lazy_load=True):
        """
        Add a single variable as a row in the tree under parent_item.
        Also handles plugin-based representation.
        """
        try:
            # Check if a plugin handler exists
            handler = None
            for data_type in type_handlers:
                if isinstance(value, data_type):
                    handler = type_handlers[data_type]
                    break

            if handler:
                # plugin-based
                representation = handler(value)
                # Type
                value_type = infer_type_hint_general(value)
                # Size (try shape from plugin or fallback)
                if representation.shape:
                    if isinstance(value, str):
                        size_str = str(len(value))
                    else:
                        size_str = ", ".join(map(str, representation.shape))
                elif self.can_expand(value):
                    size_str = self.calculate_size(value)
                else:
                    size_str = ""
                # Value (prefer plugin's value_summary)
                formatted_value = (
                    str(representation.value_summary) if representation.value_summary
                    else self.format_value(value)
                )
                # Memory usage
                memory_usage = self.format_bytes(representation.nbytes)
            else:
                # no plugin -> fallback
                value_type = infer_type_hint_general(value)
                size_str = self.calculate_size(value)
                formatted_value = self.format_value(value)
                memory_usage = self.calculate_memory_usage(value)  # Modified below

            # Create QStandardItems for each column
            item_var = QStandardItem(name)
            item_type = QStandardItem(value_type)
            item_size = QStandardItem(size_str)
            item_val = QStandardItem(formatted_value)
            item_mem = QStandardItem(memory_usage if memory_usage else "")  # Modified here

            # read-only columns
            item_var.setEditable(False)
            item_var.setFlags(item_var.flags() | Qt.ItemFlag.ItemIsDragEnabled)
            item_type.setEditable(False)
            item_size.setEditable(False)
            item_val.setEditable(False)
            item_mem.setEditable(False)

            parent_item.appendRow([item_var, item_type, item_size, item_val, item_mem])

            # If it can expand, add a placeholder row for lazy loading
            if lazy_load and self.can_expand(value):
                placeholder = QStandardItem("Loading...")
                placeholder.setEditable(False)
                item_var.appendRow([
                    placeholder, QStandardItem(), QStandardItem(), QStandardItem(), QStandardItem()
                ])
        except Exception as e:
            logger.error(f"Error adding variable '{name}': {e}")
            parent_item.appendRow([
                QStandardItem(name),
                QStandardItem("Error"),
                QStandardItem(""),
                QStandardItem(f"<Error: {e}>"),
                QStandardItem("")
            ])

    def handle_expand(self, index):
        """
        Called when a tree item is expanded; checks for a placeholder row to load children.
        """
        try:
            item = self.model.itemFromIndex(index)
            if item:
                if item.hasChildren():
                    first_child = item.child(0, 0)
                    if first_child.text() == "Loading...":
                        item.removeRow(0)
                        path = self.resolve_item_path(item)
                        value = self.resolve_variable(path)
                        if value is not None:
                            self.load_children(item, value)
                            # After loading children, resize columns to fit new content
                            self.resize_all_columns()
        except Exception as e:
            logger.error(f"Error handling expand: {e}")

    def load_children(self, parent_item, value, visited=None):
        """
        Recursively loads child attributes/elements for dictionaries, lists, or objects.
        """
        if visited is None:
            visited = set()
        try:
            if id(value) in visited:
                logger.debug(f"Cyclic reference for '{parent_item.text()}'.")
                self.add_variable("<Cyclic Reference>", "<Cyclic Reference>", parent_item, lazy_load=False)
                return
            visited.add(id(value))

            if isinstance(value, list):
                for i, elem in enumerate(value):
                    self.add_variable(f"[{i}]", elem, parent_item)
            elif isinstance(value, dict):
                for key, val in value.items():
                    self.add_variable(str(key), val, parent_item)
            elif inspect.isclass(value):
                # skip classes
                return
            elif hasattr(value, '__dict__') and not isinstance(value, QObject):
                for attr_name, attr_val in vars(value).items():
                    if attr_name.startswith("_"):
                        continue
                    if callable(attr_val):
                        continue
                    self.add_variable(attr_name, attr_val, parent_item)
            elif isinstance(value, QObject):
                # for Qt objects, reflect on public attributes
                for attr in dir(value):
                    if attr.startswith("_"):
                        continue
                    try:
                        attr_val = getattr(value, attr)
                        if callable(attr_val):
                            continue
                        self.add_variable(attr, attr_val, parent_item)
                    except Exception as sub_e:
                        logger.error(f"Error accessing {attr}: {sub_e}")
                        self.add_variable(attr, f"<Error: {sub_e}>", parent_item, lazy_load=False)
            # After loading children, adjust column sizes
            self.resize_all_columns()
        except Exception as e:
            logger.error(f"Error loading children: {e}")

    def can_expand(self, value):
        """Check if object can be expanded (dict, list, or has __dict__, but not str/bytes)."""
        return isinstance(value, (dict, list, QObject)) or (
            hasattr(value, '__dict__') and not isinstance(value, (str, bytes))
        )

    def calculate_size(self, value) -> str:
        """
        Calculate size for the "Size" column: length for strings/iterables,
        shape for shape-based objects, etc.
        """
        try:
            if isinstance(value, str):
                return str(len(value))
            elif isinstance(value, (list, tuple, set)):
                return str(len(value))
            elif isinstance(value, dict):
                return f"{{{len(value)}}}"
            elif hasattr(value, 'shape'):
                return ", ".join(map(str, value.shape))
            elif hasattr(value, '__dict__'):
                return str(len(vars(value)))
            return ""
        except Exception as e:
            logger.error(f"Error calculating size: {e}")
            return ""

    @staticmethod
    def get_element_str(obj):
        # Check if this class (or its parents) actually override __str__
        # by comparing it to the default object.__str__:
        if type(obj).__str__ != object.__str__:
            # Means there's a custom __str__ override
            return str(obj)
        else:
            # Fallback: just show the type name
            return type(obj).__name__

    def format_value(self, value) -> str:
        try:
            max_len = 150
            if isinstance(value, (list, tuple, set)):
                sample_elems = []
                for i, elem in enumerate(value):
                    if i >= 5:  # only show first 5
                        break
                    sample_elems.append(self.get_element_str(elem))
                val_str = ", ".join(sample_elems)
                val_str = f"[{val_str}] ..." if len(value) > 5 else f"[{val_str}]"
            elif isinstance(value, dict):
                first_keys = list(value.keys())[:5]
                val_str = "{" + ", ".join(map(str, first_keys)) + ("}..." if len(value) > 5 else "}")
            elif hasattr(value, '__dict__') and not isinstance(value, (str, bytes)):
                # Show a short placeholder
                val_str = f"<{type(value).__name__}>"
            else:
                val_str = str(value)

            if len(val_str) > max_len:
                return val_str[: max_len - 3] + "..."
            return val_str
        except Exception as e:
            logger.error(f"Error in format_value: {e}")
            return "<Error>"

    def calculate_memory_usage(self, value) -> str:
        """
        Calculate memory usage of a variable.
        Only display memory for built-in data types and types handled by plugins.
        For pointers or complex objects, return "".
        """
        try:
            # Define built-in types
            built_in_types = (int, float, str, bool)

            if isinstance(value, built_in_types):
                return self.format_bytes(sys.getsizeof(value))

            # Check if a plugin handles this type
            for data_type, handler in type_handlers.items():
                if isinstance(value, data_type):
                    rep = handler(value)
                    if isinstance(rep, VariableRepresentation):
                        return self.format_bytes(rep.nbytes)

            # If not built-in or handled by plugin, return ""
            return ""
        except Exception as e:
            logger.error(f"Error calculating memory usage: {e}")
            return ""

    @staticmethod
    def format_bytes(bytes_size):
        """
        Convert bytes to human-readable string.
        """
        try:
            units = ["B", "KB", "MB", "GB", "TB"]
            for u in units:
                if bytes_size < 1024:
                    if u == "B":
                        return f"{bytes_size} B"
                    return f"{bytes_size:.2f} {u}"
                bytes_size /= 1024
            return f"{bytes_size:.2f} PB"
        except Exception as e:
            logger.error(f"Error formatting bytes: {e}")
            return ""

    def show_context_menu(self, position):
        """
        Right-click context menu: Export, Update, Copy Path, etc.
        """
        indexes = self.tree_view.selectedIndexes()
        if not indexes:
            return

        menu = QMenu()

        # Single or multiple export
        if len(indexes) == 1:
            export_action = QAction("Export", self)
            export_action.triggered.connect(lambda: self.export_variable(indexes[0]))
            menu.addAction(export_action)
        else:
            export_sel_action = QAction("Export Selected", self)
            export_sel_action.triggered.connect(lambda: self.export_selected_variables(indexes))
            menu.addAction(export_sel_action)

        # Update for root-level
        for idx in indexes:
            item = self.model.itemFromIndex(idx)
            if item.parent() is None:
                update_action = QAction("Update", self)
                update_action.triggered.connect(lambda: self.update_selected_variables(indexes))
                menu.addAction(update_action)
                break

        # Copy Path action
        copy_path_action = QAction("Copy Path", self)
        copy_path_action.triggered.connect(lambda: self.copy_variable_path(indexes))
        menu.addAction(copy_path_action)

        menu.exec(self.tree_view.viewport().mapToGlobal(position))

    def export_variable(self, index):
        """
        Export a single variable.
        """
        item = self.model.itemFromIndex(index)
        var_name = item.text()
        value = self.resolve_variable(var_name)
        if value is not None:
            self.exporter.export_variable(var_name, value)
        else:
            QMessageBox.warning(
                self, "Export Warning",
                f"Variable '{var_name}' could not be resolved and was not exported."
            )

    def export_selected_variables(self, indexes):
        """
        Export multiple variables.
        """
        vars_to_export = {}
        for idx in indexes:
            if idx.column() != 0:
                continue
            item = self.model.itemFromIndex(idx)
            if item:
                path = self.resolve_item_path(item)
                val = self.resolve_variable(path)
                if val is not None:
                    vars_to_export[path] = val

        if vars_to_export:
            self.exporter.export_variables(vars_to_export)
        else:
            QMessageBox.warning(self, "Export Warning", "No valid variables selected for export.")

    def update_selected_variables(self, indexes):
        """
        Reload root-level variables for each selected item.
        """
        for idx in indexes:
            if idx.column() != 0:
                continue
            item = self.model.itemFromIndex(idx)
            if item and item.parent() is None:
                var_name = item.text()
                val = getattr(self.data_source, var_name, None)
                if val is not None:
                    self.unload_and_reload_item(item, val, var_name)
                else:
                    logger.warning(f"Root variable '{var_name}' is unavailable.")

    def unload_and_reload_item(self, item, value, path):
        """
        Clear an item's children and reload them, updating Type, Size, Value, Memory columns.
        """
        try:
            logger.debug(f"Unloading and reloading item '{path}' with value: {value}")

            # Determine if the object is 'data' or 'pointer'
            kind, _ = self.calculate_memory_usage_internal(value)

            if kind == 'data':
                # Check if a plugin handler exists
                handler = None
                for data_type in type_handlers:
                    if isinstance(value, data_type):
                        handler = type_handlers[data_type]
                        break

                if handler:
                    # plugin-based
                    representation = handler(value)
                    # Type
                    value_type = infer_type_hint_general(value)
                    # Size (try shape from plugin or fallback)
                    if representation.shape:
                        if isinstance(value, str):
                            size_str = str(len(value))
                        else:
                            size_str = ", ".join(map(str, representation.shape))
                    elif self.can_expand(value):
                        size_str = self.calculate_size(value)
                    else:
                        size_str = ""
                    # Value (prefer plugin's value_summary)
                    formatted_value = (
                        str(representation.value_summary) if representation.value_summary
                        else self.format_value(value)
                    )
                    # Memory usage
                    memory_usage = self.format_bytes(representation.nbytes)
                else:
                    # no plugin -> fallback
                    value_type = infer_type_hint_general(value)
                    size_str = self.calculate_size(value)
                    formatted_value = self.format_value(value)
                    memory_usage = self.format_bytes(sys.getsizeof(value))  # Fallback

                # Update columns
                self.model.itemFromIndex(item.index().sibling(item.row(), 1)).setText(value_type)
                self.model.itemFromIndex(item.index().sibling(item.row(), 2)).setText(size_str)
                self.model.itemFromIndex(item.index().sibling(item.row(), 3)).setText(formatted_value)
                self.model.itemFromIndex(item.index().sibling(item.row(), 4)).setText(memory_usage if memory_usage else "")
                logger.info(f"Updated columns for '{path}'.")

                # Clear existing children
                item.removeRow(0, item.rowCount())
                logger.info(f"Unloaded children for '{path}'.")

                # Reload children
                if self.can_expand(value):
                    self.load_children(item, value)
                    logger.info(f"Reloaded children for '{path}'.")
            elif kind == 'pointer':
                # For pointers or complex objects, display '' in Memory
                # Still update Type, Size, and Value
                # Type
                value_type = infer_type_hint_general(value)
                # Size
                size_str = self.calculate_size(value)
                # Value
                formatted_value = self.format_value(value)
                # Memory usage
                memory_usage = ""

                # Update columns
                self.model.itemFromIndex(item.index().sibling(item.row(), 1)).setText(value_type)
                self.model.itemFromIndex(item.index().sibling(item.row(), 2)).setText(size_str)
                self.model.itemFromIndex(item.index().sibling(item.row(), 3)).setText(formatted_value)
                self.model.itemFromIndex(item.index().sibling(item.row(), 4)).setText(memory_usage)

                logger.info(f"Updated columns for '{path}'.")

                # Clear existing children
                item.removeRow(0, item.rowCount())
                logger.info(f"Unloaded children for '{path}'.")

                # Reload children
                if self.can_expand(value):
                    self.load_children(item, value)
                    logger.info(f"Reloaded children for '{path}'")
        except Exception as e:
            logger.error(f"Error unloading/reloading '{path}': {e}")

    def calculate_memory_usage_internal(self, value) -> tuple:
        """
        Internal method to determine if the object is 'data' or 'pointer'.
        Returns a tuple ('data' or 'pointer', size_in_bytes).
        """
        built_in_types = (int, float, str, bool)
        if isinstance(value, built_in_types):
            return ('data', sys.getsizeof(value))
        for data_type in type_handlers:
            if isinstance(value, data_type):
                handler = type_handlers[data_type]
                rep = handler(value)
                if isinstance(rep, VariableRepresentation):
                    return ('data', rep.nbytes)
        return ('pointer', sys.getsizeof(value))  # Default to pointer

    def resolve_variable(self, path):
        """
        Convert a path string (like root["key"][0].child_attr) to a live object.
        """
        try:
            pattern = re.compile(r'\w+|\.\w+|\["[^"]*"\]|\[\d+\]')
            components = pattern.findall(path)
            if not components:
                return None

            root_component = components[0]
            value = getattr(self.data_source, root_component, None)
            if value is None:
                return None

            for comp in components[1:]:
                if comp.startswith("["):
                    # dict key or list index
                    if '"' in comp:
                        key = comp.strip('["]')
                        if isinstance(value, dict):
                            value = value.get(key, None)
                            if value is None:
                                return None
                        else:
                            return None
                    else:
                        idx = int(comp.strip("[]"))
                        if isinstance(value, list) and 0 <= idx < len(value):
                            value = value[idx]
                        else:
                            return None
                elif comp.startswith("."):
                    attr = comp.strip(".")
                    if hasattr(value, attr):
                        value = getattr(value, attr, None)
                    else:
                        return None
                else:
                    # unrecognized
                    return None
            return value
        except Exception as e:
            logger.error(f"Error resolving variable path '{path}': {e}")
            return None

    def resolve_item_path(self, item):
        """
        Reconstruct the path string by traversing up the parents,
        then descending into lists/dicts/attributes as needed.
        """
        parts = []
        cur = item
        while cur is not None:
            parts.append(cur.text())
            cur = cur.parent()
        parts.reverse()

        # Start from data_source
        path = ""
        value = None
        for i, part in enumerate(parts):
            if i == 0:
                path = part
                value = getattr(self.data_source, part, None)
            else:
                if isinstance(value, list):
                    path += part
                    try:
                        idx = int(part.strip("[]"))
                        value = value[idx]
                    except:
                        value = None
                elif isinstance(value, dict):
                    # escape quotes
                    escaped_key = part.replace('"', '\\"').replace("'", "\\'")
                    path += f'["{escaped_key}"]'
                    value = value.get(part, None)
                elif hasattr(value, part):
                    path += f".{part}"
                    value = getattr(value, part, None)
                else:
                    path += f".{part}"
                    value = None
        return path

    def copy_variable_path(self, indexes):
        """
        Copy the full path of the variable(s) to the clipboard.
        """
        paths = []
        for idx in indexes:
            if idx.column() != 0:
                continue
            item = self.model.itemFromIndex(idx)
            if item:
                path = self.resolve_item_path(item)
                paths.append(path)

        if paths:
            clipboard = QApplication.clipboard()
            clipboard.setText("\n".join(paths))
            logger.debug(f"Copied to clipboard: {paths}")

    def add_console(self, alias="data_source"):
        """
        Opens an IPython/Qt console in a separate window, injecting self.data_source under the given alias.
        """
        kernel_manager = QtInProcessKernelManager()
        kernel_manager.start_kernel()
        kernel_manager.kernel.gui = "qt"

        kernel_client = kernel_manager.client()
        kernel_client.start_channels()

        console = RichJupyterWidget()
        console.kernel_manager = kernel_manager
        console.kernel_client = kernel_client

        console_window = QWidget()
        console_window.setWindowTitle("Console")
        layout = QVBoxLayout(console_window)
        layout.addWidget(console)
        console_window.resize(600, 960)
        console_window.show()

        self.console_window = console_window  # keep a reference to avoid GC

        # Inject data_source
        kernel = kernel_manager.kernel.shell
        kernel.push({alias: self.data_source})

        logger.info(f"Console window opened and '{alias}' injected.")


class VariableStandardItemModel(QStandardItemModel):
    """
    Custom QStandardItemModel that supports dragging the variable path from the tree.
    """
    def __init__(self, viewer, alias="data_source", parent=None):
        super().__init__(parent)
        self.viewer = viewer
        self.alias = alias

    def supportedDragActions(self):
        return Qt.DropAction.CopyAction

    def mimeData(self, indexes):
        mime_data = super().mimeData(indexes)
        paths = []
        for idx in indexes:
            if idx.column() != 0:
                continue
            item = self.itemFromIndex(idx)
            if item:
                path = self.viewer.resolve_item_path(item)
                if path and not path.startswith(self.alias):
                    path = f"{self.alias}.{path}"
                paths.append(path)
        if paths:
            mime_data.setText("\n".join(paths))
            logger.debug(f"Dragging variable paths: {paths}")
        return mime_data

    def flags(self, index):
        if not index.isValid():
            return Qt.ItemFlag.NoItemFlags
        if index.column() == 0:
            return super().flags(index) | Qt.ItemFlag.ItemIsDragEnabled
        return super().flags(index)
