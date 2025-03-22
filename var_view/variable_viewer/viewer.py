# var_view/variable_viewer/viewer.py

import logging
import re
import sys  # Added import for sys.getsizeof

from PyQt6.QtWidgets import (
    QMainWindow, QTreeView, QVBoxLayout, QWidget, QMenu, QMessageBox,
    QHeaderView, QApplication
)
from PyQt6.QtGui import QAction, QStandardItem
from PyQt6.QtCore import Qt, QObject

from var_view.variable_exporter import VariableExporter
from var_view.plugin_manager import PluginManager  # Import PluginManager
from var_view.variable_viewer.console import ConsoleManager
from var_view.variable_viewer.model import VariableStandardItemModel
from var_view.variable_viewer.utils import infer_type_hint_general, format_bytes, \
    VariableRepresentation

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


class ObjectRef:
    __slots__ = ('obj',)

    def __init__(self, obj):
        self.obj = obj

    def __repr__(self):
        # Return a safe summary instead of traversing the object.
        return f"<ObjectRef {type(self.obj).__name__} at {hex(id(self.obj))}>"


class VariableViewer(QMainWindow):
    """
    A GUI-based viewer that:
      - Displays variables in a tree (with columns: Variable, Type, Size, Value, Memory)
      - Supports built_in_plugins for data-type-specific handling
      - Can expand nested objects or lists/dicts (lazy loading)
      - Includes a console integration for interactive usage
    """

    def __init__(self, data_source, alias="data_source", app_plugin_dir=None):
        super().__init__()
        self.data_source = data_source
        self.exporter = VariableExporter(self)

        # Initialize PluginManager
        self.plugin_manager = PluginManager()
        self.load_plugins(app_plugin_dir)

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
        self.model.setHorizontalHeaderLabels(
            ["Variable", "Type", "Size", "Value", "Memory"])
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

    def load_plugins(self, app_plugin_dir):
        """
        Load built-in and app-specific plugins.
        """
        # Load built-in plugins
        try:
            import var_view.built_in_plugins
            self.plugin_manager.load_plugins_from_package(var_view.built_in_plugins)
        except Exception as e:
            logger.error(f"Error loading built-in plugins: {e}")

        # Load app-specific plugins
        if app_plugin_dir:
            try:
                self.plugin_manager.load_plugins_from_directory(app_plugin_dir)
            except Exception as e:
                logger.error(f"Error loading app-specific plugins: {e}")

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
        self.model.setHorizontalHeaderLabels(
            ["Variable", "Type", "Size", "Value", "Memory"])

        all_vars = {
            name: getattr(self.data_source, name, None)
            for name in dir(self.data_source)
            if not name.startswith("_") and not callable(
                getattr(self.data_source, name, None))
        }

        for name, value in all_vars.items():
            self.add_variable(name, value, self.model.invisibleRootItem())

    def add_variable(self, name, value, parent_item, lazy_load=True, obj_ref=None):
        """
        Add a single variable as a row in the tree under parent_item.
        Also handles plugin-based representation and object references.

        Parameters:
        - name (str): The display name of the variable.
        - value (Any): The value of the variable.
        - parent_item (QStandardItem): The parent tree item.
        - lazy_load (bool): Whether to add a placeholder for lazy loading children.
        - obj_ref (Any): Reference to the actual object (used for object keys).
        """
        try:
            # Check if a plugin handler exists
            handler = self.plugin_manager.get_handler_for_type(value)

            if handler:
                # Plugin-based
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
                    size_str = "N/A"
                # Value (prefer plugin's value_summary)
                formatted_value = (
                    str(representation.value_summary) if representation.value_summary
                    else self.format_value(value)
                )
                # Memory usage
                memory_usage = format_bytes(representation.nbytes)
            else:
                # No plugin -> fallback
                value_type = infer_type_hint_general(value)
                size_str = self.calculate_size(value)
                formatted_value = self.format_value(value)
                memory_usage = self.calculate_memory_usage(value)

            # Create QStandardItems for each column
            item_var = QStandardItem(name)
            # Always enable drag for the variable name item
            item_var.setFlags(item_var.flags() | Qt.ItemFlag.ItemIsDragEnabled)
            item_type = QStandardItem(value_type)
            item_size = QStandardItem(size_str)
            item_val = QStandardItem(formatted_value)
            item_mem = QStandardItem(memory_usage if memory_usage else "")

            # Make columns read-only
            for item in [item_var, item_type, item_size, item_val, item_mem]:
                item.setEditable(False)

            # Determine safe drag path based on parent's safe drag text (UserRole+3)
            parent_safe = parent_item.data(Qt.ItemDataRole.UserRole + 3)
            if parent_safe:
                parent_obj = parent_item.data(Qt.ItemDataRole.UserRole + 1)
                if parent_obj is not None and (
                        isinstance(parent_obj, dict)
                        or (hasattr(parent_obj, "obj") and isinstance(parent_obj.obj,
                                                                      dict))
                ):
                    safe_drag_path = f'{parent_safe}[{repr(name)}]'
                else:
                    safe_drag_path = f'{parent_safe}.{name}'
            else:
                safe_drag_path = f"{self.model.alias}.{name}"
            item_var.setData(safe_drag_path, Qt.ItemDataRole.UserRole + 3)

            # Always store the object reference (wrapped in ObjectRef) for expandable values (except lists)
            if self.can_expand(value) and not isinstance(value, list):
                if not isinstance(obj_ref, ObjectRef):
                    obj_ref = ObjectRef(value)
                item_var.setData(obj_ref, Qt.ItemDataRole.UserRole + 1)

            parent_item.appendRow([item_var, item_type, item_size, item_val, item_mem])

            # If it can expand, add a placeholder row for lazy loading children
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

    @staticmethod
    def _build_visited_from_item(item):
        """
        Traverse up from the given item, collecting the IDs of the objects
        associated with each node (if any). This represents the visited set for
        the branch leading to this item.
        """
        visited = set()
        cur = item
        while cur is not None:
            obj = cur.data(Qt.ItemDataRole.UserRole + 1)
            if obj is not None:
                visited.add(id(obj))
            cur = cur.parent()
        return visited

    def handle_expand(self, index):
        """
        Called when a tree item is expanded; checks for a placeholder row to load children.
        """
        try:
            item = self.model.itemFromIndex(index)
            if item:
                # Retrieve the object reference if it exists
                obj_ref = item.data(Qt.ItemDataRole.UserRole + 1)
                if isinstance(obj_ref, ObjectRef):
                    value = obj_ref.obj
                else:
                    value = obj_ref
                    # Resolve the variable path as usual
                    path = self.resolve_item_path(item)
                    value = self.resolve_variable(path)

                if value is not None and item.hasChildren():
                    first_child = item.child(0, 0)
                    if first_child.text() == "Loading...":
                        item.removeRow(0)  # Remove the placeholder row
                        # Remove the placeholder and load children
                        # Build a visited set from the current branch.
                        visited = self._build_visited_from_item(item)
                        self.load_children(item, value, visited)
                        # After loading children, resize columns to fit new content
                        self.resize_all_columns()
                else:
                    logger.warning(f"No children to expand for item: {item.text()}")
            else:
                logger.warning("Item not found in model.")
        except Exception as e:
            logger.error(f"Error handling expand: {e}")

    def load_children(self, parent_item, value, visited=None):
        """
        Recursively loads child attributes/elements for dictionaries, lists, namedtuples, or objects.
        """
        if visited is None:
            visited = set()

        try:
            # Avoid cyclic references
            if id(value) in visited:
                logger.debug(f"Cyclic reference for '{parent_item.text()}'.")
                self.add_variable("<Cyclic Reference>", "<Cyclic Reference>",
                                  parent_item, lazy_load=False)
                return
            visited.add(id(value))

            # Group 1: Handle iterable-like structures
            if isinstance(value, list):
                for i, elem in enumerate(value):
                    self.add_variable(f"[{i}]", elem, parent_item)
            elif isinstance(value, dict):
                for idx, (key, val) in enumerate(value.items()):
                    if isinstance(key, (str, int, float, bool, tuple)):
                        # Basic types: display key as string
                        display_key = str(key)
                        obj_ref = val if self.can_expand(val) else None
                        self.add_variable(display_key, val, parent_item,
                                          obj_ref=obj_ref)
                    else:
                        # Object keys: display with type info
                        display_key = f"Key: {str(key)}"
                        obj_ref = val if self.can_expand(val) else None

                        # Use existing logic to add variable
                        self.add_variable(display_key, val, parent_item,
                                          obj_ref=obj_ref)

                        # Retrieve the just added item from parent_item.
                        # Assuming the new item is appended at the end:
                        row_count = parent_item.rowCount()
                        if row_count > 0:
                            # Get the last row's first column item (variable name column)
                            complex_key_item = parent_item.child(row_count - 1, 0)
                            if complex_key_item:
                                # Store (parent_dict, key_index) as metadata on this item
                                complex_key_item.setData((value, idx),
                                                         Qt.ItemDataRole.UserRole + 2)



            elif isinstance(value, tuple) and hasattr(value, '_fields'):  # named tuple
                for field in value._fields:
                    attr_val = getattr(value, field)
                    self.add_variable(field, attr_val, parent_item)
            # Group 2: Handle Qt objects
            elif isinstance(value, QObject):
                for attr in dir(value):
                    if not attr.startswith("_"):
                        try:
                            attr_val = getattr(value, attr)
                            if not callable(attr_val):
                                self.add_variable(attr, attr_val, parent_item)
                        except Exception as sub_e:
                            logger.error(f"Error accessing {attr}: {sub_e}")
                            self.add_variable(attr, f"<Error: {sub_e}>", parent_item,
                                              lazy_load=False)
            # Group 3: Handle objects with `__dict__`
            elif hasattr(value, '__dict__') and not isinstance(value, QObject):
                for attr_name, attr_val in vars(value).items():
                    if not attr_name.startswith("_") and not callable(attr_val):
                        self.add_variable(attr_name, attr_val, parent_item)
            # Group 4: Handle objects with `__slots__`
            elif hasattr(value, '__slots__'):
                for slot in value.__slots__:
                    try:
                        attr_val = getattr(value, slot)
                        self.add_variable(slot, attr_val, parent_item)
                    except AttributeError:
                        logger.debug(f"Slot {slot} not accessible.")
            # Group 5: Reflect on objects without `__dict__` or `__slots__`
            else:
                for attr in dir(value):
                    if not attr.startswith("_"):
                        try:
                            attr_val = getattr(value, attr)
                            if not callable(attr_val):
                                self.add_variable(attr, attr_val, parent_item)
                        except Exception as e:
                            logger.error(f"Error accessing {attr}: {e}")
                            self.add_variable(attr, f"<Error: {e}>", parent_item,
                                              lazy_load=False)

            # Adjust column sizes after loading children
            self.resize_all_columns()

        except Exception as e:
            logger.error(f"Error loading children: {e}")

    def can_expand(self, value):
        """Check if object can be expanded (dict, list, namedtuple, QObject, or has __dict__ or __slots__)."""
        return (
                isinstance(value, (dict, list, QObject)) or
                (isinstance(value, tuple) and hasattr(value, '_fields')) or  # named tuple
                hasattr(value, '__dict__') or
                hasattr(value, '__slots__')
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
                val_str = "{" + ", ".join(map(str, first_keys)) + (
                    "}..." if len(value) > 5 else "}")
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
        Only display memory for built-in data types and types handled by built_in_plugins.
        For pointers or complex objects, return "".
        """
        try:
            # Define built-in types
            built_in_types = (int, float, str, bool)

            if isinstance(value, built_in_types):
                return format_bytes(sys.getsizeof(value))

            # Check if a plugin handles this type
            handler = self.plugin_manager.get_handler_for_type(value)
            if handler:
                rep = handler(value)
                if isinstance(rep, VariableRepresentation):
                    return format_bytes(rep.nbytes)

            # If not built-in or handled by plugin, return ""
            return ""
        except Exception as e:
            logger.error(f"Error calculating memory usage: {e}")
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
            export_sel_action.triggered.connect(
                lambda: self.export_selected_variables(indexes))
            menu.addAction(export_sel_action)

        # Update for root-level
        for idx in indexes:
            item = self.model.itemFromIndex(idx)
            if item.parent() is None:
                update_action = QAction("Update", self)
                update_action.triggered.connect(
                    lambda: self.update_selected_variables(indexes))
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
            QMessageBox.warning(self, "Export Warning",
                                "No valid variables selected for export.")

    def update_selected_variables(self, indexes):
        """
        Reload selected variables from the data source.
        Supports both root-level and nested variables.
        """
        for idx in indexes:
            if idx.column() != 0:
                continue
            item = self.model.itemFromIndex(idx)
            if item:
                path = self.resolve_item_path(item)
                value = self.resolve_variable(path)
                if value is not None:
                    self.unload_and_reload_item(item, value, path)
                else:
                    logger.warning(f"Variable '{path}' is unavailable.")

    def unload_and_reload_item(self, item, value, path):
        """
        Clear an item's children and reload them, updating Type, Size, Value, Memory columns.
        """
        try:
            logger.debug(f"Unloading and reloading item '{path}' with value: {value}")

            # Check if a plugin handler exists
            handler = self.plugin_manager.get_handler_for_type(value)

            if handler:
                # Plugin-based
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
                    size_str = "N/A"
                # Value (prefer plugin's value_summary)
                formatted_value = (
                    str(representation.value_summary) if representation.value_summary
                    else self.format_value(value)
                )
                # Memory usage
                memory_usage = format_bytes(representation.nbytes)
            else:
                # No plugin -> fallback
                value_type = infer_type_hint_general(value)
                size_str = self.calculate_size(value)
                formatted_value = self.format_value(value)
                memory_usage = self.calculate_memory_usage(value)

            # Update columns
            self.model.itemFromIndex(item.index().sibling(item.row(), 1)).setText(
                value_type)
            self.model.itemFromIndex(item.index().sibling(item.row(), 2)).setText(
                size_str)
            self.model.itemFromIndex(item.index().sibling(item.row(), 3)).setText(
                formatted_value)
            self.model.itemFromIndex(item.index().sibling(item.row(), 4)).setText(
                memory_usage if memory_usage else "")
            logger.info(f"Updated columns for '{path}'.")

            # Clear existing children
            item.removeRows(0, item.rowCount())
            logger.info(f"Unloaded children for '{path}'.")

            # Reload children
            if self.can_expand(value):
                self.load_children(item, value)
                logger.info(f"Reloaded children for '{path}'.")
        except Exception as e:
            logger.error(f"Error unloading/reloading '{path}': {e}")

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
                    except Exception:
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

    def has_variable(self, path: str) -> bool:
        """
        Check if a variable with the given path exists in the tree model.

        Parameters:
        - path (str): The dot-separated path of the variable (e.g., "c.hi").

        Returns:
        - bool: True if the variable exists, False otherwise.
        """
        parts = path.split('.')
        current_items = self.model.invisibleRootItem()

        for part in parts:
            found = False
            for row in range(current_items.rowCount()):
                item = current_items.child(row, 0)  # Column 0: Variable Name
                if item.text() == part:
                    current_items = item
                    found = True
                    break
            if not found:
                return False
        return True

    def add_console(self, alias="data_source"):
        """
        Opens an IPython/Qt console in a separate window, injecting self.data_source under the given alias.
        """
        # Initialize ConsoleManager with a callback to refresh the view
        self.console_manager = ConsoleManager(self.data_source, alias, self)
